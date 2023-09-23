use azopt::{VRewardTree, visible_reward::{*, config::*, transitions::Transitions}, VRewardRootData, VRewardStateData};
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
    
#[derive(Clone, Debug, PartialEq, Eq)]
struct Edge(usize, usize);

#[derive(Clone, Debug, PartialEq, Eq)]
struct ColoredEdge(Edge, Color);

impl core::fmt::Display for ColoredEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let ColoredEdge(Edge(u, v), c) = self;
        match c {
            Color::Red => write!(f, "\x1b[31m{u}-{v}\x1b[0m"),
            Color::Blue => write!(f, "\x1b[34m{u}-{v}\x1b[0m"),
        }
    }
}

// impl core::fmt::Display for GraphState {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let Self { edges, time_remaining } = self;
//         write!(f, "[t = {time_remaining}]")?;
//         edges.into_iter().try_for_each(|ColoredEdge(Edge(u, v), c)| {
//             match c {
//                 Color::Red => write!(f, " \x1b[31m{u}-{v}\x1b[0m"),
//                 Color::Blue => write!(f, " \x1b[34m{u}-{v}\x1b[0m"),
//             }
//         })
//     }
// }



#[derive(Clone, Debug, PartialEq, Eq)]
struct GraphState {
    edges: Vec<ColoredEdge>,
    time_remaining: usize,
}

impl core::fmt::Display for GraphState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { edges, time_remaining } = self;
        write!(f, "[t = {time_remaining}]")?;
        edges.into_iter().try_for_each(|e| {
            write!(f, " {e}")
        })
    }
}

impl State for GraphState {
    type R = i32;

    fn is_terminal(&self) -> bool {
        self.time_remaining == 0
    }

    fn act(&mut self, action: usize) {
        let color = self.edges.get_mut(action).map(|ColoredEdge(_, c)| c).unwrap();
        *color = match color {
            Color::Red => Color::Blue,
            Color::Blue => Color::Red,
        };
        self.time_remaining -= 1;
    }

    fn cost(&self) -> Self::R {
        let mut cost = 0;
        for w in 0..5 {
            for v in 0..w {
                for u in 0..v {
                    let color_uv = self.color(u, v);
                    let color_vw = self.color(v, w);
                    let color_uw = self.color(u, w);
                    if color_uv == color_vw && color_vw == color_uw {
                        cost += 1;
                    }
                }
            }
        }
        cost
    }

    fn action_rewards(&self) -> Vec<(usize, Self::R)> {
        self.edges.iter().enumerate().map(|(i, ColoredEdge(Edge(u, v), c))| {
            let red_triangles: i32 = self.triangles(*u, *v, Color::Red);
            let blue_triangles: i32 = self.triangles(*u, *v, Color::Blue);
            // let cost = self.cost();
            // if cost == 0 && self.time_remaining == 10 {
            //     println!("{self}: i = {i}, e = {u}-{v}, (r, b) = ({red_triangles}, {blue_triangles})")
            // }
            (i, match c {
                Color::Red => red_triangles - blue_triangles,
                Color::Blue => blue_triangles - red_triangles,
            })
        }).collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Color {
    Red,
    Blue,
}

impl rand::distributions::Distribution<Color> for rand::distributions::Standard {
    fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Color {
        if rng.gen() {
            Color::Red
        } else {
            Color::Blue
        }
    }
}

impl GraphState {
    fn new(t: usize) -> Self {
        let edges = vec![
            ColoredEdge(Edge(0, 1), Color::Red),
            ColoredEdge(Edge(0, 2), Color::Red),
            ColoredEdge(Edge(1, 2), Color::Red),
            ColoredEdge(Edge(0, 3), Color::Red),
            ColoredEdge(Edge(1, 3), Color::Red),
            ColoredEdge(Edge(2, 3), Color::Red),
            ColoredEdge(Edge(0, 4), Color::Red),
            ColoredEdge(Edge(1, 4), Color::Red),
            ColoredEdge(Edge(2, 4), Color::Red),
            ColoredEdge(Edge(3, 4), Color::Red),
        ];
        Self { edges, time_remaining: t }
    }

    fn generate_random<R: rand::Rng>(t: usize, rng: &mut R) -> Self {
        let mut edges = Vec::new();
        for j in 0..5 {
            for i in 0..j {
                edges.push(ColoredEdge(Edge(i, j), rng.gen()));
            }
        }
        Self { edges, time_remaining: t }
    }

    // fn neighborhoods(&self, u: usize) -> (Vec<usize>, Vec<usize>) {
    //     let mut red_neighborhood = Vec::new();
    //     let mut blue_neighborhood = Vec::new();
    //     for (edge, color) in &self.edges {
    //         match edge {
    //             Edge(w, x) if *w == u => {
    //                 match color {
    //                     Color::Red => red_neighborhood.push(*x),
    //                     Color::Blue => blue_neighborhood.push(*x),
    //                 }
    //             }
    //             Edge(w, x) if *x == u => {
    //                 match color {
    //                     Color::Red => red_neighborhood.push(*w),
    //                     Color::Blue => blue_neighborhood.push(*w),
    //                 }
    //             }
    //             _ => {}
    //         }
    //     }
    //     (red_neighborhood, blue_neighborhood)
    // }

    fn color(&self, u: usize, v: usize) -> Color {
        let (u, v) = if u < v { (u, v) } else { (v, u) };
        self.edges.iter().find(|ColoredEdge(Edge(a, b), _)| u.eq(a) && v.eq(b)).expect(&format!("{u},{v}")).1
    }

    fn triangles(&self, u: usize, v: usize, color: Color) -> i32 {
        // let c = self.cost();
        let mut triangles = 0;
        for w in 0..5 {
            if w == u || w == v {
                continue;
            }
            let color_uw = self.color(u, w);
            let color_vw = self.color(v, w);
            if color_uw == color_vw && color_vw == color {
                // if c == 0 && self.time_remaining == 10 {
                //     match color {
                //         Color::Red => println!("{self}: \x1b[31m{u}-{v}-{w}\x1b[0m, "),
                //         Color::Blue => println!("{self}: \x1b[34m{u}-{v}-{w}\x1b[0m, "),
                //     }
                // }
                triangles += 1;
            }
        }
        triangles
    }
}

#[derive(Debug, Default, PartialOrd, Ord, PartialEq, Eq, Clone)]
struct GraphPath {
    actions: Vec<usize>,
}

impl Path for GraphPath {
    fn new(action: usize) -> Self {
        Self { actions: vec![action] }
    }

    fn push(&mut self, action: usize) {
        self.actions.push(action);
    }
}

struct GraphPrediction;
struct GraphModel;
impl GraphModel {
    fn new() -> Self {
        Self
    }
}

impl Model<GraphState> for GraphModel {
    type P = GraphPrediction;
    fn predict(&self, state: &GraphState) -> Self::P {
        GraphPrediction
    }
}

// struct GraphRootData;
type GraphRootData = VRewardRootData!(GraphConfig);
// struct GraphStateData;
type GraphStateData = VRewardStateData!(GraphConfig);

impl Prediction<GraphStateData, GraphRootData> for GraphPrediction {
    type G = f32;
    type R = i32;

    fn value(&self) -> &Self::G {
        &0.0
    }

    fn new_data(&self, transitions: Vec<(usize, Self::R)>) -> (GraphStateData, Self::G) {
        (GraphStateData::new(transitions.into_iter().map(|(i, r)| {
            (i, r, 0.1)
        }).collect()), 0.0)
    }

    fn new_root_data(&self, cost: Self::R, transitions: Vec<(usize, Self::R)>) -> GraphRootData {
        GraphRootData::new(cost, transitions.into_iter().map(|(i, r)| {
            (i, r, 0.1)
        }).collect())
    }
}

// impl Prediction<GraphStateData> for GraphPrediction {
//     type G = f32;
//     fn new_data(&self) -> (GraphStateData, Self::G) {
//         todo!()
//     }
// }

// impl SortedActions for GraphRootData {
//     type R = i32;
//     type G = f32;
//     fn best_action(&self) -> (usize, i32) {
//         todo!()
//     }

//     fn update_future_reward(&mut self, action: usize, reward: &Self::R) {
//         todo!()
//     }

//     fn update_futured_reward_and_expected_gain(&mut self, action: usize, reward: &Self::R, gain: &Self::G) {
//         todo!()
//     }
// }

// impl SortedActions for GraphStateData {
//     type R = i32;
//     type G = f32;
//     fn best_action(&self) -> (usize, i32) {
//         todo!()
//     }

//     fn update_future_reward(&mut self, action: usize, reward: &Self::R) {
//         todo!()
//     }

//     fn update_futured_reward_and_expected_gain(&mut self, action: usize, reward: &Self::R, gain: &Self::G) {
//         todo!()
//     }
// }

struct GraphConfig;

impl Config for GraphConfig {
    type RootData = GraphRootData;
    type StateData = GraphStateData;
    type Prediction = GraphPrediction;
    type Path = GraphPath;
    type State = GraphState;
    type Model = GraphModel;
    type Reward = i32;
    type ExpectedFutureGain = f32;
}

type VSTree = VRewardTree!(GraphConfig);
// type VSTree = <GraphConfig as ToVisibleRewardTree>::VRewardTree;

#[derive(Default)]
struct BasicGraphLog {
    graph: String,
    root_cost: i32,
    data: Vec<BasicGraphLogData>,
}

struct BasicGraphLogData {
    pub transitions: Vec<(usize, i32)>,
    pub end: BasicGraphLogEnd,
}

enum BasicGraphLogEnd {
    Leaf,
    New(f32),
}

impl BasicGraphLogData {
    fn new<'a, I>(a1: usize, r1: i32, next_transitions: I, end: BasicGraphLogEnd) -> Self
    where
        I: Iterator<Item = (&'a usize, &'a i32)>,
    {
        let mut transitions = vec![(a1, r1)];
        transitions.extend(next_transitions.map(|(a, r)| (*a, *r)));
        Self { transitions, end }
    }
}

impl BasicGraphLog {
    fn update(&mut self, data: BasicGraphLogData) {
        self.data.push(data);
    }

    fn new(state: &GraphState, pred: &GraphPrediction) -> Self {
        Self { graph: state.to_string(), root_cost: state.cost(), data: Default::default() }
    }
}

impl core::fmt::Display for BasicGraphLog {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { graph, data, root_cost } = self;
        data.into_iter().try_for_each(|data| {
            write!(f, "{graph} ({root_cost}){data}")
        })
    }
}

impl core::fmt::Display for BasicGraphLogData {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { transitions, end } = self;
        transitions.into_iter().try_for_each(|(a, r)| {
            const E: &[&str] = &["0-1", "0-2", "1-2", "0-3", "1-3", "2-3", "0-4", "1-4", "2-4", "3-4"];
            write!(f, " -> {} ({r}) ->", E[*a])
        })?;
        match end {
            BasicGraphLogEnd::Leaf => write!(f, " Leaf"),
            BasicGraphLogEnd::New(v) => write!(f, " New({v})"),
        }
    }
}

fn main() {
    let mut model = GraphModel::new();
    let trees = 
        (0..16)
        .into_par_iter()
        .map(|_| GraphState::generate_random(10, &mut rand::thread_rng()))
        .map(|state| {
            let prediction = model.predict(&state);
            let log = BasicGraphLog::new(&state, &prediction);
            let tree = VSTree::new::<GraphConfig>(state, prediction);
            (tree, log)
        });
    let trees = trees
        .map(|(tree, log)| (tree.simulate_once::<GraphConfig>(), tree, log))
        .map(|(transitions, mut tree, mut log)| {
            let Transitions { a1, r1, transitions, end } = transitions;
            match end {
                transitions::FinalState::Leaf(_, _) => {
                    let trans = transitions.iter().map(|(_, a, r)| (a, r));
                    let data = BasicGraphLogData::new(a1, r1, trans, BasicGraphLogEnd::Leaf);
                    log.update(data);
                    tree.update_with_transitions::<GraphConfig>(a1, transitions);
                },
                transitions::FinalState::New(p, s) => {
                    let prediction = model.predict(&s);
                    let end = BasicGraphLogEnd::New(*prediction.value());
                    let data = BasicGraphLogData::new(a1, r1, transitions.iter().map(|(_, a, r)| (a, r)), end);
                    log.update(data);
                    let t = s.action_rewards();
                    let eval = tree.insert::<GraphConfig>(p, t, prediction);
                    tree.update_with_transitions_and_evaluation::<GraphConfig>(a1, transitions, eval)
                },
            }
            (tree, log)
        }).collect::<Vec<_>>();
    for (_, log) in trees {
        println!("{log}")
    }
}
