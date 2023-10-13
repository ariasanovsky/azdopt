// use az_discrete_opt::{
//     ir_tree::{
//         config::*,
//         log::{FinalStateData, Log},
//         model::Model,
//         stats::{VRewardRootData, VRewardStateData},
//         *,
//     },
//     VRewardTree,
// };
// use rayon::prelude::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

// #[derive(Clone, Debug, PartialEq, Eq)]
// struct Edge(usize, usize);

// #[derive(Clone, Debug, PartialEq, Eq)]
// struct ColoredEdge(Edge, Color);

// impl core::fmt::Display for ColoredEdge {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let ColoredEdge(Edge(u, v), c) = self;
//         match c {
//             Color::Red => write!(f, "\x1b[31m{u}-{v}\x1b[0m"),
//             Color::Blue => write!(f, "\x1b[34m{u}-{v}\x1b[0m"),
//         }
//     }
// }

// #[derive(Clone, Debug, PartialEq, Eq)]
// struct GraphState {
//     edges: Vec<ColoredEdge>,
//     time_remaining: usize,
// }

// impl core::fmt::Display for GraphState {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let Self {
//             edges,
//             time_remaining,
//         } = self;
//         write!(f, "[t = {time_remaining}]")?;
//         edges.into_iter().try_for_each(|e| write!(f, " {e}"))
//     }
// }

// impl State for GraphState {
//     fn is_terminal(&self) -> bool {
//         self.time_remaining == 0
//     }

//     fn act(&mut self, action: usize) {
//         let color = self
//             .edges
//             .get_mut(action)
//             .map(|ColoredEdge(_, c)| c)
//             .unwrap();
//         *color = match color {
//             Color::Red => Color::Blue,
//             Color::Blue => Color::Red,
//         };
//         self.time_remaining -= 1;
//     }

//     fn cost(&self) -> f32 {
//         let mut cost = 0;
//         for w in 0..5 {
//             for v in 0..w {
//                 for u in 0..v {
//                     let color_uv = self.color(u, v);
//                     let color_vw = self.color(v, w);
//                     let color_uw = self.color(u, w);
//                     if color_uv == color_vw && color_vw == color_uw {
//                         cost += 1;
//                     }
//                 }
//             }
//         }
//         cost as f32
//     }

//     fn action_rewards(&self) -> Vec<(usize, f32)> {
//         self.edges
//             .iter()
//             .enumerate()
//             .map(|(i, ColoredEdge(Edge(u, v), c))| {
//                 let red_triangles: i32 = self.triangles(*u, *v, Color::Red);
//                 let blue_triangles: i32 = self.triangles(*u, *v, Color::Blue);
//                 // let cost = self.cost();
//                 // if cost == 0 && self.time_remaining == 10 {
//                 //     println!("{self}: i = {i}, e = {u}-{v}, (r, b) = ({red_triangles}, {blue_triangles})")
//                 // }
//                 (
//                     i,
//                     match c {
//                         Color::Red => red_triangles - blue_triangles,
//                         Color::Blue => blue_triangles - red_triangles,
//                     } as f32,
//                 )
//             })
//             .collect()
//     }
// }

// #[derive(Clone, Copy, Debug, PartialEq, Eq)]
// enum Color {
//     Red,
//     Blue,
// }

// impl rand::distributions::Distribution<Color> for rand::distributions::Standard {
//     fn sample<R: rand::Rng + ?Sized>(&self, rng: &mut R) -> Color {
//         if rng.gen() {
//             Color::Red
//         } else {
//             Color::Blue
//         }
//     }
// }

// impl GraphState {
//     fn generate_random<R: rand::Rng>(t: usize, rng: &mut R) -> Self {
//         let mut edges = Vec::new();
//         for j in 0..5 {
//             for i in 0..j {
//                 edges.push(ColoredEdge(Edge(i, j), rng.gen()));
//             }
//         }
//         Self {
//             edges,
//             time_remaining: t,
//         }
//     }

//     fn color(&self, u: usize, v: usize) -> Color {
//         let (u, v) = if u < v { (u, v) } else { (v, u) };
//         self.edges
//             .iter()
//             .find(|ColoredEdge(Edge(a, b), _)| u.eq(a) && v.eq(b))
//             .expect(&format!("{u},{v}"))
//             .1
//     }

//     fn triangles(&self, u: usize, v: usize, color: Color) -> i32 {
//         let mut triangles = 0;
//         for w in 0..5 {
//             if w == u || w == v {
//                 continue;
//             }
//             let color_uw = self.color(u, w);
//             let color_vw = self.color(v, w);
//             if color_uw == color_vw && color_vw == color {
//                 triangles += 1;
//             }
//         }
//         triangles
//     }
// }

// #[derive(Debug, Default, PartialOrd, Ord, PartialEq, Eq, Clone)]
// struct GraphPath {
//     actions: Vec<usize>,
// }

// impl Path for GraphPath {
//     fn new(action: usize) -> Self {
//         Self {
//             actions: vec![action],
//         }
//     }

//     fn push(&mut self, action: usize) {
//         self.actions.push(action);
//     }
// }

// struct GraphPrediction;
// struct GraphModel;
// impl GraphModel {
//     fn new() -> Self {
//         Self
//     }
// }

// impl Model<GraphState> for GraphModel {
//     type P = GraphPrediction;
//     type O = (GraphState, (Vec<(usize, f32)>, f32));
//     type L = ();

//     fn predict(&self, _: &GraphState) -> Self::P {
//         GraphPrediction
//     }

//     fn update(&mut self, _: Vec<Self::O>) -> Self::L {}
// }

// type GraphRootData = VRewardRootData;
// type GraphStateData = VRewardStateData;

// impl Prediction<GraphStateData, GraphRootData> for GraphPrediction {
//     fn value(&self) -> &f32 {
//         &0.0
//     }

//     fn new_data(&self, transitions: Vec<(usize, f32)>) -> (GraphStateData, f32) {
//         (
//             GraphStateData::new(transitions.into_iter().map(|(i, r)| (i, r, 0.1)).collect()),
//             0.0,
//         )
//     }

//     fn new_root_data(&self, cost: f32, transitions: Vec<(usize, f32)>) -> GraphRootData {
//         GraphRootData::new(
//             cost,
//             transitions.into_iter().map(|(i, r)| (i, r, 0.1)).collect(),
//         )
//     }
// }
// struct GraphConfig;

// impl Config for GraphConfig {
//     type RootData = GraphRootData;
//     type StateData = GraphStateData;
//     type Prediction = GraphPrediction;
//     type Path = GraphPath;
//     type State = GraphState;
//     type Model = GraphModel;
//     type Log = BasicGraphLog;
// }

// type VSTree = VRewardTree!(GraphConfig);

// #[derive(Default)]
// struct BasicGraphLog {
//     root: String,
//     root_cost: f32,
//     data: Vec<BasicGraphLogData>,
// }

// struct BasicGraphLogData {
//     pub actions_and_costs: Vec<(usize, f32)>,
//     pub end: FinalStateData,
// }

// impl BasicGraphLogData {
//     fn new<'a, I>(
//         root_cost: f32,
//         a1: usize,
//         r1: f32,
//         next_transitions: I,
//         end: FinalStateData,
//     ) -> Self
//     where
//         I: Iterator<Item = (&'a usize, &'a f32)>,
//     {
//         let mut cost = root_cost;
//         cost += r1;
//         let mut actions_and_costs = vec![(a1, cost)];
//         actions_and_costs.extend(next_transitions.map(|(a, r)| {
//             cost += r;
//             (*a, cost)
//         }));
//         Self {
//             actions_and_costs,
//             end,
//         }
//     }
// }

// impl BasicGraphLog {
//     fn update(&mut self, data: BasicGraphLogData) {
//         self.data.push(data);
//     }

//     fn new(root: &GraphState, _: &GraphPrediction) -> Self {
//         Self {
//             root: root.to_string(),
//             root_cost: root.cost(),
//             data: Default::default(),
//         }
//     }
// }

// impl core::fmt::Display for BasicGraphLog {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let Self {
//             root: graph,
//             data,
//             root_cost,
//         } = self;
//         data.into_iter()
//             .try_for_each(|data| writeln!(f, "{graph} ({root_cost}){data}"))
//     }
// }

// impl core::fmt::Display for BasicGraphLogData {
//     fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
//         let Self {
//             actions_and_costs: transitions,
//             end,
//         } = self;
//         transitions.into_iter().try_for_each(|(a, r)| {
//             const E: &[&str] = &[
//                 "0-1", "0-2", "1-2", "0-3", "1-3", "2-3", "0-4", "1-4", "2-4", "3-4",
//             ];
//             write!(f, " -> {} ({r})", E[*a])
//         })?;
//         match end {
//             FinalStateData::Leaf => write!(f, " -> Leaf"),
//             FinalStateData::New { final_reward } => write!(f, " -> New({final_reward})"),
//             // FinalStateData:Leaf => write!(f, " -> Leaf"),
//             // FinalState::New(_, _) => write!(f, " -> New"),
//             // BasicGraphLogEnd::Leaf => write!(f, " -> Leaf"),
//             // BasicGraphLogEnd::New(v) => write!(f, " -> New({v})"),
//         }
//     }
// }

// impl Log for BasicGraphLog {
//     type T = Vec<(GraphPath, usize, f32)>;
//     fn add_transition_data(
//         &mut self,
//         a1: usize,
//         r1: f32,
//         transition: &Self::T,
//         end: log::FinalStateData,
//     ) {
//         let transitions = transition.iter().map(|(_, a, r)| (a, r));
//         let data = BasicGraphLogData::new(self.root_cost, a1, r1, transitions, end);
//         self.update(data);
//     }
// }

// fn main() {
//     let mut model = GraphModel::new();
//     let mut trees: Vec<_> = (0..16)
//         .into_par_iter()
//         .map(|_| GraphState::generate_random(10, &mut rand::thread_rng()))
//         .map(|state| {
//             let prediction = model.predict(&state);
//             let log = BasicGraphLog::new(&state, &prediction);
//             let tree = VSTree::new::<GraphConfig>(state, prediction);
//             (tree, log)
//         })
//         .collect();
//     (0..20).for_each(|_| {
//         trees.par_iter_mut().for_each(|(tree, ref mut log)| {
//             tree.simulate_once_and_update::<GraphConfig>(&model, log);
//         });
//     });
//     let observations: Vec<_> = trees
//         .into_iter()
//         .map(|(tree, log)| {
//             println!("{log}");
//             let o = tree.to_observation();
//             // dbg!(&o);
//             o
//         })
//         .collect();
//     model.update(observations);
// }
fn main() {
    
}