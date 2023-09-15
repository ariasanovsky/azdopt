use azopt::{VisibleRewardTree, SearchPath};

#[derive(Clone)]
struct R33State;

impl R33State {
    fn new() -> Self {
        Self
    }
}

#[derive(Default, PartialOrd, Ord, PartialEq, Eq)]
struct R33Path;

impl SearchPath for R33Path {}

type R33Reward = i32;

type R33FutureReward = i32;

const A_TOTAL: usize = 10;

struct R33Evaluate;

impl azopt::Evaluate<R33State> for R33Evaluate {
    fn evaluate(&self, state: &R33State) -> (Vec<f32>, f32) {
        todo!()
    }
}

impl azopt::State<R33Reward> for R33State {
    fn reward(&self, action: usize) -> Option<R33Reward> {
        todo!()
    }
}

fn main() {
    let root = R33State::new();
    let eval = R33Evaluate;
    let mut tree = VisibleRewardTree::<
        R33State,
        R33Path,
        R33Reward,
        R33FutureReward,
        A_TOTAL,
    >::new::<R33Evaluate>(root, &eval);
    let sims: usize = 100;
    let depth: usize = 10;
    tree.simulate(sims, depth, &eval);
    let best = tree.best();
}
