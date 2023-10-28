use std::collections::BTreeMap;
// use core::num::NonZeroUsize;
use crate::{iq_min_tree::ActionsTaken, state::Cost};

pub struct INTMinTree {
    root_data: INTStateData,
    data: BTreeMap<ActionsTaken, INTStateData>,
}

impl INTMinTree {
    pub fn new<S: INTState>(root_predictions: &[f32], cost: f32, root: &S) -> Self {
        Self {
            root_data: INTStateData::new(root_predictions, cost, root),
            data: BTreeMap::new(),
        }
    }

    pub fn replant<S: INTState>(&mut self, root_predictions: &[f32], cost: f32, root: &S) {
        self.data.clear();
        self.root_data = INTStateData::new(root_predictions, cost, root);
    }

    pub fn simulate_once<S: INTState>(&self, state: &mut S, state_vec: &mut [f32]) -> INTTransitions {
        let Self { root_data, data } = self;
        let first_action = root_data.best_action();
        state.act(first_action);
        let mut state_path = ActionsTaken::new(first_action);
        let mut transitions: Vec<(ActionsTaken, f32, usize)> = vec![];
        let mut cost = root_data.cost;
        while !state.is_terminal() {
            if let Some(data) = data.get(&state_path) {
                let action = data.best_action();
                state.act(action);
                state_path.push(action);
                cost = data.cost;
                transitions.push((state_path.clone(), cost, action));
            } else {
                state.update_vec(state_vec);
                return INTTransitions {
                    first_action,
                    transitions,
                    end: INTSearchEnd::Unvisited { state_path },
                }
            }
        }
        INTTransitions {
            first_action,
            transitions,
            end: INTSearchEnd::Terminal { state_path, cost },
        }
    }

    pub fn insert(
        &mut self,
        transitions: &INTTransitions,
        probabilities: &[f32],
    ) {
        let Self { root_data, data } = self;
        todo!()
    }
}

pub trait INTState {
    fn act(&mut self, action: usize);
    fn is_terminal(&self) -> bool;
    fn update_vec(&self, state_vec: &mut [f32]);
    fn actions(&self) -> Vec<usize>;
}

pub struct INTTransitions {
    pub(crate) first_action: usize,
    pub(crate) transitions: Vec<(ActionsTaken, f32, usize)>,
    pub(crate) end: INTSearchEnd,
}

impl INTTransitions {
    pub fn last_cost(&self) -> Option<f32> {
        match &self.end {
            INTSearchEnd::Terminal { cost, .. } => Some(*cost),
            INTSearchEnd::Unvisited { .. } => None,
        }
    }

    pub fn last_path(&self) -> &ActionsTaken {
        match &self.end {
            INTSearchEnd::Terminal { state_path, .. } => state_path,
            INTSearchEnd::Unvisited { state_path, .. } => state_path,
        }
    }
}

pub(crate) enum INTSearchEnd {
    Terminal { state_path: ActionsTaken, cost: f32 },
    Unvisited { state_path: ActionsTaken },
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
    pub fn new<S: INTState>(predctions: &[f32], cost: f32, state: &S) -> Self {
        let mut actions = state.actions().into_iter().map(|a| {
            let p = predctions[a];
            INTActionData::new(a, p)
        }).collect::<Vec<_>>();
        actions.sort_by(|a, b| b.upper_estimate.total_cmp(&a.upper_estimate));
        Self {
            frequency: 0,
            cost,
            actions,
        }
    }

    pub fn best_action(&self) -> usize {
        let Self {
            frequency: _,
            cost: _,
            actions,
        } = self;
        actions[0].action
    }
}

struct INTActionData {
    action: usize,
    probability: f32,
    frequency: usize,
    q_sum: f32,
    upper_estimate: f32,    
}

const C_PUCT_0: f32 = 1.0;
const C_PUCT: f32 = 1.0;

impl INTActionData {
    pub fn new(action: usize, probability: f32) -> Self {
        Self {
            action,
            probability,
            frequency: 0,
            q_sum: 0.0,
            upper_estimate: C_PUCT_0 * probability,
        }
    }
}