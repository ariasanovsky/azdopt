use azopt::{VisibleRewardTree, visible_tree::{*, config::*}};
use rand::Rng;
use rayon::prelude::{IntoParallelIterator, ParallelIterator};
    
#[derive(Clone, Debug, PartialEq, Eq)]
struct Edge(usize, usize);

#[derive(Clone, Debug, PartialEq, Eq)]
struct GraphState {
    edges: Vec<(Edge, Color)>,
    time_remaining: usize,
}

impl State for GraphState {
    fn is_terminal(&self) -> bool {
        todo!()
    }

    fn act(&mut self, action: usize) {
        todo!()
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
            (Edge(0, 1), Color::Red),
            (Edge(0, 2), Color::Red),
            (Edge(1, 2), Color::Red),
            (Edge(0, 3), Color::Red),
            (Edge(1, 3), Color::Red),
            (Edge(2, 3), Color::Red),
            (Edge(0, 4), Color::Red),
            (Edge(1, 4), Color::Red),
            (Edge(2, 4), Color::Red),
            (Edge(3, 4), Color::Red),
        ];
        Self { edges, time_remaining: t }
    }

    fn generate_random<R: rand::Rng>(t: usize, rng: &mut R) -> Self {
        let mut edges = Vec::new();
        for j in 0..5 {
            for i in 0..j {
                edges.push((Edge(i, j), rng.gen()));
            }
        }
        Self { edges, time_remaining: t }
    }

    fn neighborhoods(&self, u: usize) -> (Vec<usize>, Vec<usize>) {
        let mut red_neighborhood = Vec::new();
        let mut blue_neighborhood = Vec::new();
        for (edge, color) in &self.edges {
            match edge {
                Edge(w, x) if *w == u => {
                    match color {
                        Color::Red => red_neighborhood.push(*x),
                        Color::Blue => blue_neighborhood.push(*x),
                    }
                }
                Edge(w, x) if *x == u => {
                    match color {
                        Color::Red => red_neighborhood.push(*w),
                        Color::Blue => blue_neighborhood.push(*w),
                    }
                }
                _ => {}
            }
        }
        (red_neighborhood, blue_neighborhood)
    }
}

#[derive(Default, PartialOrd, Ord, PartialEq, Eq, Clone)]
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

impl Model<GraphState, GraphPrediction> for GraphModel {
    fn predict(&self, state: &GraphState) -> GraphPrediction {
        GraphPrediction
    }
}

struct GraphRootData;

impl Prediction<GraphRootData> for GraphPrediction {
    type G = f32;
    fn new_data(&self) -> (GraphRootData, Self::G) {
        todo!()
    }
}

struct GraphStateData;
impl Prediction<GraphStateData> for GraphPrediction {
    type G = f32;
    fn new_data(&self) -> (GraphStateData, Self::G) {
        todo!()
    }
}

impl SortedActions for GraphRootData {
    type R = i32;
    type G = f32;
    fn best_action(&self) -> (usize, i32) {
        todo!()
    }

    fn update_future_reward(&mut self, action: usize, reward: &Self::R) {
        todo!()
    }

    fn update_futured_reward_and_expected_gain(&mut self, action: usize, reward: &Self::R, gain: &Self::G) {
        todo!()
    }
}

impl SortedActions for GraphStateData {
    type R = i32;
    type G = f32;
    fn best_action(&self) -> (usize, i32) {
        todo!()
    }

    fn update_future_reward(&mut self, action: usize, reward: &Self::R) {
        todo!()
    }

    fn update_futured_reward_and_expected_gain(&mut self, action: usize, reward: &Self::R, gain: &Self::G) {
        todo!()
    }
}

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

type VSTree = VisibleRewardTree!(GraphConfig);
// type VSTree = <GraphConfig as ToVisibleRewardTree>::VRewardTree;

fn main() {
    let mut model = GraphModel::new();
    let trees = 
        (0..16)
        .into_par_iter()
        .map(|_| GraphState::generate_random(10, &mut rand::thread_rng()))
        .map(|state| {
            let prediction = model.predict(&state);
            VSTree::new::<GraphConfig>(state, prediction)
        });
    let trees = trees
        .map(|tree| (tree.simulate_once::<GraphConfig>(), tree))
        .map(|((first_action, transitions, final_state), mut tree)| {
            match final_state {
                FinalState::Leaf => {
                    tree.update_with_transitions::<GraphConfig>(first_action, transitions);
                },
                FinalState::New(path, s) => {
                    let prediction = model.predict(&s);
                    let eval = tree.insert::<GraphConfig>(path, prediction);
                    tree.update_with_transitions_and_evaluation::<GraphConfig>(first_action, transitions, eval);
                },
            }
            tree
        }).collect::<Vec<_>>();
}