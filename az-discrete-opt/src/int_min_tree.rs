use std::collections::BTreeMap;
// use core::num::NonZeroUsize;
use crate::{iq_min_tree::{ActionsTaken, Transitions}, state::Cost};

pub struct INTMinTree {
    root_data: INTStateData,
    data: BTreeMap<ActionsTaken, INTStateData>,
}

impl INTMinTree {
    pub fn new(root_predictions: &[f32], cost: f32) -> Self {
        Self {
            root_data: INTStateData::new(root_predictions, cost),
            data: BTreeMap::new(),
        }
    }

    pub fn replant(&mut self, root_predictions: &[f32], cost: f32) {
        self.data.clear();
        self.root_data = INTStateData::new(root_predictions, cost);
    }

    pub fn simulate_once<S: INTState + Cost>(&self, state: &mut S, state_vec: &mut [f32]) -> INTTransitions {
        let Self { root_data, data } = self;
        let first_action = root_data.best_action();
        state.act(state_vec, first_action);
        let mut state_path = ActionsTaken::new(first_action);
        let mut transitions: Vec<(ActionsTaken, f32, usize)> = vec![];
        while !state.is_terminal() {
            if let Some(data) = data.get(&state_path) {
                let action = data.best_action();
                state.act(state_vec, action);
                state_path.push(action);
                transitions.push((state_path.clone(), data.cost, action));
            } else {
                return INTTransitions {
                    first_action,
                    transitions,
                    end: INTSearchEnd::Unvisited { state_path, cost: state.cost() },
                }
            }
        }
        INTTransitions {
            first_action,
            transitions,
            end: INTSearchEnd::Terminal { state_path, cost: state.cost() },
        }
    }
}

pub trait INTState {
    fn act(&mut self, state_vec: &mut [f32], action: usize);
    fn is_terminal(&self) -> bool;
}

pub struct INTTransitions {
    first_action: usize,
    transitions: Vec<(ActionsTaken, f32, usize)>,
    end: INTSearchEnd,
}

enum INTSearchEnd {
    Terminal { state_path: ActionsTaken, cost: f32 },
    Unvisited { state_path: ActionsTaken, cost: f32 },
}

/* todo! refactor so that:
    `actions` = [a0, ..., a_{k-1}, a_k, ..., a_{n-1}]
        here, a0, ..., a_{k-1} are visited, the rest unvisited
        when initialized, we sort by probability
        when an action is visited for the first time, we increment the counter k
        visited actions are sorted by upper estimate
        when selecting the next action, we compare the best visited action to the best unvisited action
        unvisited actions use the same upper estimate formula, but it depends only on the probability
*/
struct INTStateData {
    frequency: usize,
    cost: f32,
    actions: Vec<INTActionData>,
}

impl INTStateData {
    pub fn new(predctions: &[f32], cost: f32) -> Self {
        Self {
            frequency: 0,
            cost,
            actions: predctions.iter().enumerate().map(|(a, p)| INTActionData::new(a, *p)).collect(),
        }
    }

    pub fn best_action(&self) -> usize {
        todo!()
    }
}

struct INTActionData {
    action: usize,
    probability: f32,
    frequency: usize,
    q_sum: f32,
    upper_estimate: f32,    
}

impl INTActionData {
    pub fn new(action: usize, probability: f32) -> Self {
        Self {
            action,
            probability,
            frequency: 0,
            q_sum: 0.0,
            upper_estimate: 0.0,
        }
    }
}