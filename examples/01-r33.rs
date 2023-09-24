// use azopt::{VisibleRewardTree, Path};

// #[derive(Clone, Debug, PartialEq, Eq)]
// struct R33Edge(usize, usize);

// #[derive(Clone, Debug, PartialEq, Eq)]
// struct R33State {
//     edges: Vec<(R33Edge, R33Color)>,
//     time_remaining: usize,
// }

// #[derive(Clone, Copy, Debug, PartialEq, Eq)]
// enum R33Color {
//     Red,
//     Blue,
// }

// impl R33State {
//     fn new(t: usize) -> Self {
//         let edges = vec![
//             (R33Edge(0, 1), R33Color::Red),
//             (R33Edge(0, 2), R33Color::Red),
//             (R33Edge(1, 2), R33Color::Red),
//             (R33Edge(0, 3), R33Color::Red),
//             (R33Edge(1, 3), R33Color::Red),
//             (R33Edge(2, 3), R33Color::Red),
//             (R33Edge(0, 4), R33Color::Red),
//             (R33Edge(1, 4), R33Color::Red),
//             (R33Edge(2, 4), R33Color::Red),
//             (R33Edge(3, 4), R33Color::Red),
//         ];
//         Self { edges, time_remaining: t }
//     }

//     fn neighborhoods(&self, u: usize) -> (Vec<usize>, Vec<usize>) {
//         let mut red_neighborhood = Vec::new();
//         let mut blue_neighborhood = Vec::new();
//         for (edge, color) in &self.edges {
//             match edge {
//                 R33Edge(w, x) if *w == u => {
//                     match color {
//                         R33Color::Red => red_neighborhood.push(*x),
//                         R33Color::Blue => blue_neighborhood.push(*x),
//                     }
//                 }
//                 R33Edge(w, x) if *x == u => {
//                     match color {
//                         R33Color::Red => red_neighborhood.push(*w),
//                         R33Color::Blue => blue_neighborhood.push(*w),
//                     }
//                 }
//                 _ => {}
//             }
//         }
//         (red_neighborhood, blue_neighborhood)
//     }
// }

// #[derive(Default, PartialOrd, Ord, PartialEq, Eq, Clone)]
// struct R33Path {
//     actions: Vec<usize>,
// }

// impl Path for R33Path {
//     fn add_action(&mut self, action: usize) {
//         self.actions.push(action);
//     }
// }

// type R33Reward = i32;

// type R33FutureReward = f32;

// const A_TOTAL: usize = 10;

// struct R33Evaluate;

// impl azopt::Evaluate<R33State> for R33Evaluate {
//     fn evaluate(&self, _state: &R33State) -> (Vec<f32>, f32) {
//         (vec![0.1; A_TOTAL], 0.0)
//     }
// }

// impl azopt::State for R33State {
//     type R = R33Reward;
//     fn reward(&self, action: usize) -> Option<R33Reward> {
//         let (edge, old_color) = self.edges.get(action).unwrap();
//         let (u, v) = (edge.0, edge.1);
//         let u_neighborhoods = self.neighborhoods(u);
//         let v_neighborhoods = self.neighborhoods(v);
//         let red_uv_triangles: usize = u_neighborhoods.0.iter().filter(|&w| v_neighborhoods.0.contains(w)).count();
//         let blue_uv_triangles: usize = u_neighborhoods.1.iter().filter(|&w| v_neighborhoods.1.contains(w)).count();
//         match old_color {
//             R33Color::Red => Some((red_uv_triangles as i32) - (blue_uv_triangles as i32)),
//             R33Color::Blue => Some((blue_uv_triangles as i32) - (red_uv_triangles as i32)),
//         }
//     }

//     fn is_terminal(&self) -> bool {
//         self.time_remaining == 0
//     }

//     fn act(&mut self, action: usize) {
//         let edge = self.edges.get_mut(action).unwrap();
//         match edge.1 {
//             R33Color::Red => edge.1 = R33Color::Blue,
//             R33Color::Blue => edge.1 = R33Color::Red,
//         }
//         self.time_remaining -= 1;
//     }
// }

// struct R33SearchConfig;
// impl azopt::VisibleRewardSearchConfig for R33SearchConfig {
//     type S = R33State;
//     type P = R33Path;
//     type F = R33FutureReward;
// }

// struct R33ModelConfig;
// impl azopt::VisibleRewardModelConfig for R33ModelConfig {}

// fn main() {
//     use azopt::State;
//     let total_time = 5;
//     let root = R33State::new(total_time);
//     dbg!(root.reward(0));
//     let eval = R33Evaluate;
//     let mut tree = VisibleRewardTree::<R33SearchConfig, R33ModelConfig>::new(root);
//     let sims: usize = 100_000;
//     tree.simulate_and_update::<_, A_TOTAL>(sims, &eval);
//     let best = tree.best();
// }

fn main() {
    
}