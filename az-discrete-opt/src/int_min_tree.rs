use std::collections::BTreeMap;
// use core::num::NonZeroUsize;
use crate::iq_min_tree::ActionsTaken;

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

    pub fn simulate_once<S>(&self, state: &mut S, actions: &mut [f32]) -> Transitions {
        todo!()
    }
}

pub struct Transitions;

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