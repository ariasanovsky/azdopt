use std::collections::BTreeMap;

pub mod config;

use config::*;

pub struct VRewardTree<S, P, D0, D> {
    root: S,
    root_data: D0,
    data: BTreeMap<P, D>
}

impl<S, P, D0, D> VRewardTree<S, P, D0, D> {
    pub fn new<C>(root: S, root_prediction: C::P) -> Self
    where
        C: HasPrediction, 
        C::P: Prediction<D0>,
    {
        let root_data = root_prediction.new_data();
        Self {
            root,
            root_data,
            data: BTreeMap::new()
        }
    }
    
    pub fn simulate_once<C>(&self) -> (Vec<(usize, C::R, P)>, FinalState<P, S>)
    where
        S: Clone + State,
        D0: SortedActions<C::R>,
        C: HasReward,
        P: Path + Clone + Ord,
        D: SortedActions<C::R>,
    {
        let mut state = self.root.clone();
        let first_transition = self.root_data.best_action();
        state.act(first_transition.0);
        let mut state_path = P::new(first_transition.0);
        let mut transitions = vec![];
        while !state.is_terminal() {
            if let Some(data) = self.data.get(&state_path) {
                let (action, reward) = data.best_action();
                state.act(action);
                transitions.push((action, reward, state_path.clone()));
                state_path.push(action);
            } else {
                // new state
                return (transitions, FinalState::New(state_path, state));
            }
        }
        // terminal state
        (transitions, FinalState::Leaf)
    }
}

pub trait Prediction<D> {
    fn new_data(&self) -> D;
}

// pub trait Log {

// }

pub trait SortedActions<R> {
    fn best_action(&self) -> (usize, R);
}

pub trait Model<S, P> {
    fn predict(&self, state: &S) -> P;
}

pub trait State {
    fn is_terminal(&self) -> bool;
    fn act(&mut self, action: usize);
}

pub trait Path {
    fn new(action: usize) -> Self;
    fn push(&mut self, action: usize);
}

pub enum FinalState<P, S> {
    Leaf,
    New(P, S),
}