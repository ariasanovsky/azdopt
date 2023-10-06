use core::mem::MaybeUninit;
use core::mem::transmute;

use azopt::ir_tree::ir_min_tree::IRMinTree;
use azopt::ir_tree::ir_min_tree::IRState;
use azopt::ir_tree::transitions::Transitions;
use bit_iter::BitIter;
use dfdx::optim::Adam;
use dfdx::prelude::{Linear, ReLU, DeviceBuildExt, Module};
use dfdx::tensor::{AutoDevice, TensorFrom, AsArray};
use dfdx::tensor_ops::{AdamConfig, WeightDecay};
use priority_queue::PriorityQueue;
use ramsey::{ColoredCompleteGraph, MulticoloredGraphEdges, MulticoloredGraphNeighborhoods, OrderedEdgeRecolorings, CliqueCounts, C, E, Color, N, EdgeRecoloring};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator};

const ACTION: usize = C * E;
const BATCH: usize = 64;

#[derive(Clone, Debug)]
struct GraphState {
    colors: ColoredCompleteGraph,
    edges: MulticoloredGraphEdges,
    neighborhoods: MulticoloredGraphNeighborhoods,
    available_actions: [[bool; E]; C],
    ordered_actions: OrderedEdgeRecolorings,
    counts: CliqueCounts,
    time_remaining: usize,
}

impl GraphState {
    fn generate_random<R: rand::Rng>(t: usize, rng: &mut R) -> Self {
        let mut edges: [[bool; E]; C] = [[false; E]; C];
        let mut neighborhoods: [[u32; N]; C] = [[0; N]; C];
        let mut colors: [MaybeUninit<Color>; E] = unsafe {
            let colors: MaybeUninit<[Color; E]> = MaybeUninit::uninit();
            transmute(colors)
        };
        let mut available_actions: [[bool; E]; C] = [[true; E]; C];
        let edge_iterator = (0..N).map(|v| (0..v).map(move |u| (u, v))).flatten();
        edge_iterator.zip(colors.iter_mut()).enumerate().for_each(|(i, ((u, v), color))| {
            let c = rng.gen_range(0..C);
            edges[c][i] = true;
            available_actions[c][i] = false;
            neighborhoods[c][u] |= 1 << v;
            neighborhoods[c][v] |= 1 << u;
            color.write(Color(c));
        });
        let colors: [Color; E] = unsafe {
            transmute(colors)
        };
        let mut counts: [[MaybeUninit<i32>; E]; C] = unsafe {
            let counts: MaybeUninit<[[i32; E]; C]> = MaybeUninit::uninit();
            transmute(counts)
        };
        neighborhoods.iter().zip(counts.iter_mut()).for_each(|(neighborhoods, counts)| {
            let edge_iterator = (0..N).map(|v| (0..v).map(move |u| (u, v))).flatten();
            edge_iterator.zip(counts.iter_mut()).for_each(|((u, v), k)| {
                let neighborhood = neighborhoods[u] & neighborhoods[v];
                let count = BitIter::from(neighborhood).map(|w| {
                    (neighborhood & neighborhoods[w]).count_ones()
                }).sum::<u32>() / 2;
                k.write(count as i32);
            });
        });
        let counts: [[i32; E]; C] = unsafe {
            transmute(counts)
        };
        let mut recolorings: PriorityQueue<EdgeRecoloring, i32> = PriorityQueue::new();
        colors.iter().enumerate().for_each(|(i, c)| {
            let old_color = c.0;
            let old_count = counts[old_color][i];
            (0..C).filter(|c| old_color.ne(c)).for_each(|new_color| {
                let new_count = counts[new_color][i];
                let reward = old_count - new_count;
                let recoloring = EdgeRecoloring { new_color, edge_position: i };
                recolorings.push(recoloring, reward);
            })
        });
        Self {
            colors: ColoredCompleteGraph(colors),
            edges: MulticoloredGraphEdges(edges),
            neighborhoods: MulticoloredGraphNeighborhoods(neighborhoods),
            available_actions,
            ordered_actions: OrderedEdgeRecolorings(recolorings),
            counts: CliqueCounts(counts),
            time_remaining: t,
        }
    }

    fn generate_batch(t: usize) -> [Self; BATCH] {
        let mut states: [MaybeUninit<Self>; BATCH] = unsafe {
            MaybeUninit::uninit().assume_init()
        };
        states.par_iter_mut().for_each(|s| {
            s.write(GraphState::generate_random(t, &mut rand::thread_rng()));
        });
        unsafe {
            transmute(states)
        }
    }

    fn generate_vecs(states: &[Self; BATCH]) -> [StateVec; BATCH] {
        let mut state_vecs: [MaybeUninit<StateVec>; BATCH] = unsafe {
            MaybeUninit::uninit().assume_init()
        };
        states.par_iter().zip(state_vecs.par_iter_mut()).for_each(|(s, v)| {
            v.write(s.to_vec());
        });
        unsafe {
            transmute(state_vecs)
        }
    }

    fn to_vec(&self) -> StateVec {
        let Self {
            colors: ColoredCompleteGraph(colors),
            edges: MulticoloredGraphEdges(edges),
            neighborhoods: _,
            available_actions,
            ordered_actions: OrderedEdgeRecolorings(ordered_actions),
            counts: CliqueCounts(counts),
            time_remaining,
        } = self;
        let edge_iter = edges.iter().flatten().map(|b| {
            if *b {
                1.0
            } else {
                0.0
            }
        });
        let count_iter = counts.iter().flatten().map(|c| *c as f32);
        let action_iter = available_actions.iter().flatten().map(|a| {
            if *a {
                1.0
            } else {
                0.0
            }
        });
        let time_iter = Some(*time_remaining as f32).into_iter();
        let state_iter = edge_iter.chain(count_iter).chain(action_iter).chain(time_iter);
        let mut state_vec: StateVec = [0.0; STATE];
        state_vec.iter_mut().zip(state_iter).for_each(|(v, s)| {
            *v = s;
        });
        state_vec
    }
}

impl IRState for GraphState {
    fn cost(&self) -> f32 {
        let Self {
            colors: ColoredCompleteGraph(colors),
            edges: _,
            neighborhoods: _,
            available_actions: _,
            ordered_actions: _,
            counts: CliqueCounts(counts),
            time_remaining: _,
        } = self;
        let count = colors.iter().enumerate().map(|(i, c)| {
            let Color(c) = c;
            counts[*c][i]
        }).sum::<i32>();
        count as f32
    }

    fn action_rewards(&self) -> Vec<(usize, f32)> {
        let Self {
            colors: ColoredCompleteGraph(colors),
            edges: _,
            neighborhoods: _,
            available_actions: _,
            ordered_actions: OrderedEdgeRecolorings(ordered_actions),
            counts: _,
            time_remaining: _,
        } = self;
        ordered_actions.iter().map(|(recolor, reward)| {
            let EdgeRecoloring { new_color, edge_position } = recolor;
            let action_index = new_color * E + edge_position;
            (action_index, *reward as f32)
        }).collect()
    }
}

const STATE: usize = 6 * E + 1;
type StateVec = [f32; STATE];
type PredictionVec = [f32; PREDICTION];
type Tree = IRMinTree<GraphState>;

fn plant_forest(states: &[GraphState; BATCH], predictions: &[PredictionVec; BATCH]) -> [Tree; BATCH] {
    let mut trees: [MaybeUninit<Tree>; BATCH] = unsafe {
        let trees: MaybeUninit<[Tree; BATCH]> = MaybeUninit::uninit();
        transmute(trees)
    };
    trees.par_iter_mut().zip_eq(states.par_iter().zip_eq(predictions.par_iter())).for_each(|(t, (s, p))| {
        t.write(IRMinTree::new(s, p));
    });
    unsafe {
        transmute(trees)
    }
}

fn main() {
    let states: [GraphState; BATCH] = GraphState::generate_batch(20);
    let state_vecs: [StateVec; BATCH] = GraphState::generate_vecs(&states);
    dbg!(state_vecs.get(0).unwrap());
    
    // set up model
    let dev = AutoDevice::default();
    let mut model = dev.build_module::<Architecture, f32>();
    // let mut opt = Adam::new(
    //     &model,
    //     AdamConfig {
    //         lr: 1e-2,
    //         betas: [0.5, 0.25],
    //         eps: 1e-6,
    //         weight_decay: Some(WeightDecay::Decoupled(1e-2)),
    //      }
    // );

    let state_tensor = dev.tensor(state_vecs);
    let mut prediction_tensor = model.forward(state_tensor);
    let mut predictions: [PredictionVec; BATCH] = prediction_tensor.array();
    let mut trees: [Tree; BATCH] = plant_forest(&states, &predictions);

    const EPOCH: usize = 1;
    
    (0..EPOCH).for_each(|epoch| {
        const EPISODES: usize = 1;
        (0..EPISODES).for_each(|episode| {
            let transitions: [Trans; BATCH] = simulate_forest_once(&trees);
            todo!()
        });
    });
    todo!()
}

fn simulate_forest_once(trees: &[Tree; BATCH]) -> [Trans; BATCH] {
    let transitions: [MaybeUninit<Trans>; BATCH] = unsafe {
        let transitions: MaybeUninit<[Trans; BATCH]> = MaybeUninit::uninit();
        transmute(transitions)
    };
    todo!()
}

struct GraphPath;
type Trans = Transitions<GraphPath, GraphState>;

const HIDDEN_1: usize = 256;
const HIDDEN_2: usize = 128;
const PREDICTION: usize = ACTION + 1;

type Architecture = (
    (Linear<STATE, HIDDEN_1>, ReLU),
    (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
    Linear<HIDDEN_2, PREDICTION>,
);
