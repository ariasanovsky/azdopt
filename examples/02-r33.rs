use azopt::{VisibleRewardTree, visible_tree::{*, config::*}};
    
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
    fn new_data(&self) -> GraphRootData {
        todo!()
    }
}

struct GraphStateData;
impl Prediction<GraphStateData> for GraphPrediction {
    fn new_data(&self) -> GraphStateData {
        todo!()
    }
}

impl SortedActions<i32> for GraphRootData {
    fn best_action(&self) -> (usize, i32) {
        todo!()
    }
}

impl SortedActions<i32> for GraphStateData {
    fn best_action(&self) -> (usize, i32) {
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
}
// impl_config!(GraphConfig);

type VSTree = VisibleRewardTree!(GraphConfig);

fn main() {
    let mut model = GraphModel::new();
    let root = GraphState::new(10);
    let root_prediction = model.predict(&root);
    let mut tree: VSTree = VSTree::new::<GraphConfig>(root, root_prediction);
    tree.simulate_once::<GraphConfig>();
}