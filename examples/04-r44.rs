use core::mem::MaybeUninit;
use core::mem::transmute;

use azopt::ir_tree::ir_min_tree::IRMinTree;
use azopt::ir_tree::ir_min_tree::IRState;
use azopt::ir_tree::ir_min_tree::Transitions;
use bit_iter::BitIter;
use dfdx::optim::Adam;
use dfdx::prelude::Optimizer;
use dfdx::prelude::ZeroGrads;
use dfdx::prelude::cross_entropy_with_logits_loss;
use dfdx::prelude::mse_loss;
use dfdx::prelude::{Linear, ReLU, DeviceBuildExt, Module};
use dfdx::shapes::HasShape;
use dfdx::tensor::Trace;
use dfdx::tensor::{AutoDevice, TensorFrom, AsArray};
use dfdx::tensor_ops::Backward;
use dfdx::tensor_ops::{AdamConfig, WeightDecay};
use itertools::Itertools;
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

    fn act(&mut self, action: usize) {
        let Self {
            colors: ColoredCompleteGraph(colors),
            edges: MulticoloredGraphEdges(edges),
            neighborhoods: MulticoloredGraphNeighborhoods(neighborhoods),
            available_actions,
            ordered_actions: OrderedEdgeRecolorings(ordered_actions),
            counts: CliqueCounts(counts),
            time_remaining,
        } = self;
        let new_uv_color = action / E;
        let edge_position = action % E;
        let Color(old_uv_color) = colors[edge_position];
        let (u, v) = edge_from_position(edge_position);
        /* when does k(xy, c) change?
            necessarily, c is either old_color or new_color
            k(uv, c) never changes
            k(uw, c) and k(vw, c) may change
            k(wx, c) may also change
        */

        /* when does k = k(uw, c_old) change? (w != u, v)
            k counts the number of quadruples Q = {u, w, x, y} s.t.
                G_{c_old}[Q] + {uw} is a clique
            with this recoloring, Q is no longer a clique iff
                Q = {u, w, v, x} for some x
                w is in N_{c_old}(v) \ {u}
                x is in N_{c_old}(u) & N_{c_old}(w) & (N_{c_old}(v) \ {u})
                since x is not in N_c(u), we may omit the `\ {u}`'s if we wish
        */
        // after updating the counts, we update all affected values of r(xy, c)
        // r(uv, c) is unaffected (in fact, these values are removed)
        // we store which edges wx have an affected column and update them later
        // todo!() this can be optimized by only updating the affected columns within the iterator
        let mut affected_count_columns: Vec<usize> = vec![];

        let old_neigh_v = neighborhoods[old_uv_color][v];
        let old_neigh_u = neighborhoods[old_uv_color][u];
        let old_neigh_uv = old_neigh_u & old_neigh_v;
        
        BitIter::from(old_neigh_v).for_each(|w| {
            let old_neigh_w = neighborhoods[old_uv_color][w];
            let old_neigh_uvw = old_neigh_uv & old_neigh_w;
            let k_uw_old_decrease = old_neigh_uvw.count_ones();
            if k_uw_old_decrease != 0 {
                // decrease k(uw, c_uv_old)
                let uw_position = edge_to_position(u, w);
                counts[old_uv_color][uw_position] -= k_uw_old_decrease as i32;
                /* when does r = r(e, c) change?
                    consider r(e, c) = k(e, c_e) - k(e, c)
                        here, c_e is the current color of e
                        r(e, c) is only defined when c_e != c
                    w.l.o.g., e = uw
                    Case I: c_uw = c_uv_old
                        r(uw, c) = k(uw, c_uv_old) - k(uw, c)
                        assumes that c != c_uv_old
                        so r(uw, c) deecreases by the same decrease to k(uw, c_uv_old)
                    Case II: c_uw
                        todo!("adjust all affected values of r(uw, c)")
                */
                affected_count_columns.push(uw_position);
            }
        });
        BitIter::from(old_neigh_u).for_each(|w| {
            let old_neigh_w = neighborhoods[old_uv_color][w];
            let old_neigh_uvw = old_neigh_uv & old_neigh_w;
            let k_vw_old_decrease = old_neigh_uvw.count_ones();
            if k_vw_old_decrease != 0 {
                // decrease k(vw, c_old)
                let vw_position = edge_to_position(v, w);
                counts[old_uv_color][vw_position] -= k_vw_old_decrease as i32;
                // todo!("adjust all affected values of r(vw, c)")
                affected_count_columns.push(vw_position);
            }
        });
        /* when does k = k(wx, c_old) change? (w, x != u, v)
            k counts the number of quadruples Q = {w, x, u', v'} s.t.
                G_{c_old}[Q] + {wx} is a clique
            with this recoloring, Q is no longer a clique iff
                Q = {w, x, u, v}
                w, x are in N_{c_old}(u) & N_{c_old}(v)
        */
        BitIter::from(old_neigh_uv).tuple_combinations().for_each(|(w, x)| {
            let wx_position = edge_to_position(w, x);
            counts[old_uv_color][wx_position] -= 1;
            // todo!("adjust all affected values of r(wx, c)")
            affected_count_columns.push(wx_position);
        });

        let new_neigh_v = neighborhoods[new_uv_color][v];
        let new_neigh_u = neighborhoods[new_uv_color][u];
        let new_neigh_uv = new_neigh_u & new_neigh_v;
        
        BitIter::from(new_neigh_v).for_each(|w| {
            let new_neigh_w = neighborhoods[new_uv_color][w];
            let new_neigh_uvw = new_neigh_uv & new_neigh_w;
            let k_uw_new_increase = new_neigh_uvw.count_ones();
            if k_uw_new_increase != 0 {
                // decrease k(uw, c_uv_old)
                let uw_position = edge_to_position(u, w);
                counts[new_uv_color][uw_position] -= k_uw_new_increase as i32;
                /* when does r = r(e, c) change?
                    consider r(e, c) = k(e, c_e) - k(e, c)
                        here, c_e is the current color of e
                        r(e, c) is only defined when c_e != c
                    w.l.o.g., e = uw
                    Case I: c_uw = c_uv_old
                        r(uw, c) = k(uw, c_uv_old) - k(uw, c)
                        assumes that c != c_uv_old
                        so r(uw, c) deecreases by the same decrease to k(uw, c_uv_old)
                    Case II: c_uw
                        todo!("adjust all affected values of r(uw, c)")
                */
                affected_count_columns.push(uw_position);
            }
        });
        BitIter::from(new_neigh_u).for_each(|w| {
            let new_neigh_w = neighborhoods[new_uv_color][w];
            let new_neigh_uvw = new_neigh_uv & new_neigh_w;
            let k_vw_new_increase = new_neigh_uvw.count_ones();
            if k_vw_new_increase != 0 {
                // decrease k(vw, c_old)
                let vw_position = edge_to_position(v, w);
                counts[new_uv_color][vw_position] -= k_vw_new_increase as i32;
                // todo!("adjust all affected values of r(vw, c)")
                affected_count_columns.push(vw_position);
            }
        });
        /* when does k = k(wx, c_old) change? (w, x != u, v)
            k counts the number of quadruples Q = {w, x, u', v'} s.t.
                G_{c_old}[Q] + {wx} is a clique
            with this recoloring, Q is no longer a clique iff
                Q = {w, x, u, v}
                w, x are in N_{c_old}(u) & N_{c_old}(v)
        */
        BitIter::from(new_neigh_uv).tuple_combinations().for_each(|(w, x)| {
            let wx_position = edge_to_position(w, x);
            counts[new_uv_color][wx_position] -= 1;
            // todo!("adjust all affected values of r(wx, c)")
            affected_count_columns.push(wx_position);
        });

        affected_count_columns.into_iter().for_each(|wx_position| {
            let Color(wx_color) = colors[wx_position];
            let old_count = counts[wx_color][wx_position];
            // update r(wx, c) for all c != wx_color
            (0..C).for_each(|c| {
                let reward = old_count - counts[c][wx_position];
                let recoloring = EdgeRecoloring { new_color: c, edge_position: wx_position };
                let reward = ordered_actions.change_priority(&recoloring, reward);
                // todo!("update all affected count columns");
            });
        });

        // todo!("update colors");
        colors[edge_position] = Color(new_uv_color);
        // todo!("update edges");
        edges[old_uv_color][edge_position] = false;
        edges[new_uv_color][edge_position] = true;
        // todo!("update neighborhoods");
        neighborhoods[old_uv_color][u] &= !(1 << v);
        neighborhoods[old_uv_color][v] &= !(1 << u);
        neighborhoods[new_uv_color][u] |= 1 << v;
        neighborhoods[new_uv_color][v] |= 1 << u;
        // todo!("update available_actions");
        (0..C).for_each(|c| {
            available_actions[c][edge_position] = false;
        });
        // todo!("update ordered_actions");
        // todo!("remove uv actions from ordered_actions");
        (0..C).for_each(|c| {
            let recoloring = EdgeRecoloring { new_color: c, edge_position };
            ordered_actions.remove(&recoloring);
        });
        // todo!("update counts");
        // todo!("update time_remaining");
        *time_remaining -= 1;
    }

    fn is_terminal(&self) -> bool {
        let Self {
            colors: _,
            edges: _,
            neighborhoods: _,
            available_actions: _,
            ordered_actions: _,
            counts: _,
            time_remaining,
        } = self;
        *time_remaining == 0
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
    const EPOCH: usize = 1;
    const EPISODES: usize = 1;
    
    // set up model
    let dev = AutoDevice::default();
    let mut model = dev.build_module::<Architecture, f32>();
    let mut opt = Adam::new(
        &model,
        AdamConfig {
            lr: 1e-2,
            betas: [0.5, 0.25],
            eps: 1e-6,
            weight_decay: Some(WeightDecay::Decoupled(1e-2)),
         }
    );
    
    (0..EPOCH).for_each(|epoch| {
        let mut grads = model.alloc_grads();
        let roots: [GraphState; BATCH] = GraphState::generate_batch(20);
        let root_vecs: [StateVec; BATCH] = GraphState::generate_vecs(&roots);
        dbg!(root_vecs.get(0).unwrap());
        
        let root_tensor = dev.tensor(root_vecs.clone());
        let mut prediction_tensor = model.forward(root_tensor);
        let mut predictions: [PredictionVec; BATCH] = prediction_tensor.array();
        let mut trees: [Tree; BATCH] = plant_forest(&roots, &predictions);

        
        // play episodes
        (0..EPISODES).for_each(|episode| {
            let (transitions, end_states): ([Trans; BATCH], [GraphState; BATCH]) = simulate_forest_once(&trees);
            let end_state_vecs: [StateVec; BATCH] = state_batch_to_vecs(&end_states);
            prediction_tensor = model.forward(dev.tensor(end_state_vecs));
            predictions = prediction_tensor.array();
            update_forest(&mut trees, &transitions, &predictions);
        });
        // backprop loss
        let observations: [PredictionVec; BATCH] = forest_observations(&trees);
        grads = {
            let root_tensor = dev.tensor(root_vecs.clone());
            let traced_predictions = model.forward(root_tensor.trace(grads));
            let predicted_logits = traced_predictions.slice((0.., 0..ACTION));
            assert_eq!(predicted_logits.shape(), &(BATCH, ACTION));
            let observed_probabilities = dev.tensor(observations.clone());
            let observed_probabilities_tensor = observed_probabilities.clone().slice((0.., 0..ACTION));
            assert_eq!(predicted_logits.shape(), &(BATCH, ACTION));
            let cross_entropy = cross_entropy_with_logits_loss(predicted_logits, observed_probabilities_tensor);
            print!("{:10}\t", cross_entropy.array());
            cross_entropy.backward()
        };
        grads = {
            let root_tensor = dev.tensor(root_vecs.clone());
            let traced_predictions = model.forward(root_tensor.trace(grads));
            let predicted_values = traced_predictions.slice((0.., ACTION..));
            assert_eq!(predicted_values.shape(), &(BATCH, 1));

            let target_values = [[10.0f32; 1]; BATCH];
            let target_values = dev.tensor(target_values.clone()).slice((0.., 0..));

            let square_error = mse_loss(predicted_values, target_values);
            println!("{}", square_error.array());
            square_error.backward()
        };
        opt.update(&mut model, &grads).unwrap();
        model.zero_grads(&mut grads);
    });
    todo!()
}

fn forest_observations(trees: &[Tree; BATCH]) -> [PredictionVec; BATCH] {
    todo!()
}

fn update_forest(
    trees: &mut [Tree; BATCH],
    transitions: &[Trans; BATCH],
    predictions: &[PredictionVec; BATCH]
) {
    trees.par_iter_mut().zip_eq(transitions.par_iter()).zip_eq(predictions.par_iter()).for_each(|((tree, trans), pred)| {
        tree.update(trans, pred);
    });
}

fn state_batch_to_vecs(states: &[GraphState; BATCH]) -> [StateVec; BATCH] {
    let mut state_vecs: [MaybeUninit<StateVec>; BATCH] = unsafe {
        let state_vecs: MaybeUninit<[StateVec; BATCH]> = MaybeUninit::uninit();
        transmute(state_vecs)
    };
    state_vecs.par_iter_mut().zip_eq(states.par_iter()).for_each(|(v, s)| {
        v.write(s.to_vec());
    });
    unsafe {
        transmute(state_vecs)
    }
}

fn simulate_forest_once(trees: &[Tree; BATCH]) -> ([Trans; BATCH], [GraphState; BATCH]) {
    let mut transitions: [MaybeUninit<Trans>; BATCH] = unsafe {
        let transitions: MaybeUninit<[Trans; BATCH]> = MaybeUninit::uninit();
        transmute(transitions)
    };
    let mut state_vecs: [MaybeUninit<GraphState>; BATCH] = unsafe {
        let state_vecs: MaybeUninit<[GraphState; BATCH]> = MaybeUninit::uninit();
        transmute(state_vecs)
    };
    trees.par_iter().zip_eq(transitions.par_iter_mut()).zip_eq(state_vecs.par_iter_mut()).for_each(|((tree, t), s)| {
        let (trans, state) = tree.simulate_once();
        t.write(trans);
        s.write(state);
    });
    unsafe {
        (transmute(transitions), transmute(state_vecs))
    }
}

type Trans = Transitions;

const HIDDEN_1: usize = 256;
const HIDDEN_2: usize = 128;
const PREDICTION: usize = ACTION + 1;

type Architecture = (
    (Linear<STATE, HIDDEN_1>, ReLU),
    (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
    Linear<HIDDEN_2, PREDICTION>,
);

/* edges are enumerated in colex order:
    [0] 01  [1] 02  [3] 03
            [2] 12  [4] 13
                    [5] 23
*/
fn edge_from_position(position: usize) -> (usize, usize) {
    /* note the positions
        {0, 1}: 0 = (2 choose 2) - 1
        {1, 2}: 2 = (3 choose 2) - 1    increased by 2
        {2, 3}: 5 = (4 choose 2) - 1    increased by 3
        {3, 4}: 9 = (5 choose 2) - 1    increased by 4
        ...
        {v-1, v}: (v+1 choose 2) - 1
    */
    // the smart thing is a lookup table or direct computation
    /* solve for v from position
         (v+1 choose 2) - 1 = position
         8 * (v+1 choose 2) = 8 * position + 8
         (2*v + 1)^2 = 8 * position + 9
            2*v + 1 = sqrt(8 * position + 9)
            v = (sqrt(8 * position + 9) - 1) / 2
        etc
    */
    // todo!() we do a lazy linear search
    let mut v = 1;
    let mut upper_position = 0;
    while upper_position < position {
        v += 1;
        upper_position += v;
    }
    let difference = upper_position - position;
    let u = v - difference;
    (u, v)
}

fn edge_to_position(u: usize, v: usize) -> usize {
    let (u, v) = if u < v {
        (u, v)
    } else {
        (v, u)
    };
    /* note the positions
        {0, 1}: 0 = (2 choose 2) - 1
        {1, 2}: 2 = (3 choose 2) - 1
        {2, 3}: 5 = (4 choose 2) - 1
        {3, 4}: 9 = (5 choose 2) - 1
        ...
        {v-1, v}: (v+1 choose 2) - 1
    */
    let upper_position = v * (v + 1) / 2;
    // subtract the difference between u & v
    let difference = v - u;
    upper_position - difference
}