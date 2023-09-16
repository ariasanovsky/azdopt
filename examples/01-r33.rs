use std::collections::HashSet;

use azopt::{VisibleRewardTree, Path};

#[derive(Clone)]
struct R33State {
    edges: HashSet<(usize, usize)>,
    time_remaining: usize,
}

impl R33State {
    fn new(t: usize) -> Self {
        Self { edges: Default::default(), time_remaining: t }
    }
}

#[derive(Default, PartialOrd, Ord, PartialEq, Eq, Clone)]
struct R33Path;

impl Path for R33Path {
    fn add_action(&mut self, action: usize) {
        todo!()
    }
}

type R33Reward = i32;

type R33FutureReward = f32;

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

    fn is_terminal(&self) -> bool {
        todo!()
    }

    fn act(&mut self, action: usize) {
        todo!()
    }
}

fn main() {
    let total_time = 10;
    let root = R33State::new(total_time);
    let eval = R33Evaluate;
    let mut tree = VisibleRewardTree::<
        R33State,
        R33Path,
        R33Reward,
        R33FutureReward,
        A_TOTAL,
    >::new(root);
    let sims: usize = 100;
    tree.simulate_and_update(sims, &eval);
    let best = tree.best();
}
