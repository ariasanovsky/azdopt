use std::collections::BTreeMap;

use crate::ir_tree::stats::SortedActions;

use super::stats::{VRewardRootData, VRewardStateData, VRewardActionData};

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct ActionsTaken {
    actions_taken: Vec<usize>,
}

impl ActionsTaken {
    pub fn new(first_action: usize) -> Self {
        Self {
            actions_taken: vec![first_action],
        }
    }

    pub fn push(&mut self, action: usize) {
        self.actions_taken.push(action);
        self.actions_taken.sort_unstable();
    }
}

pub struct IRMinTree<S> {
    root: S,
    root_data: VRewardRootData,
    data: BTreeMap<ActionsTaken, VRewardStateData>
}

impl<S> IRMinTree<S> {
    pub fn new(root: &S, probability_predictions: &[f32]) -> Self
    where
        S: Clone + IRState,
    {
        let rewards = root.action_rewards();
        let actions = rewards.into_iter().map(|(i, r)| {
            let p = *probability_predictions.get(i).unwrap();
            let data = VRewardActionData {
                action: i,
                frequency: 0,
                probability: p,
                reward: r,
                future_reward_sum: 0.0,
            };
            const C_PUCT: f32 = 1.0;
            let u = r + C_PUCT * p;
            (data, u)
        }).collect();
        let root_data = VRewardRootData {
            cost: root.cost(),
            frequency: 0,
            actions,
        };
        Self {
            root: root.clone(),
            root_data,
            data: Default::default(),
        }
    }

    pub fn simulate_once(&self) -> (Transitions, S)
    where
        S: Clone + IRState,
    {
        let Self { root, root_data, data } = self;
        let mut state = root.clone();
        let (first_action, first_reward) = root_data.best_action();
        state.act(first_action);
        let mut state_path = ActionsTaken::new(first_action);
        let mut transitions: Vec<(ActionsTaken, usize, f32)> = vec![];
        while !state.is_terminal() {
            if let Some(data) = data.get(&state_path) {
                let (action, reward) = data.best_action();
                state.act(action);
                transitions.push((state_path.clone(), action, reward));
                state_path.push(action);
            } else {
                // new state
                return (Transitions {
                    first_action,
                    first_reward,
                    transitions,
                    reached_terminal: false,
                }, state);
            }
        }
        (
            Transitions {
                first_action,
                first_reward,
                transitions,
                reached_terminal: true,
            },
            state,
        )
    }

    pub fn update(&mut self, transitions: &Transitions, prediction: &[f32]) {
        todo!()
    }
}

pub trait IRState {
    fn cost(&self) -> f32;
    fn action_rewards(&self) -> Vec<(usize, f32)>;
    fn act(&mut self, action: usize);
    fn is_terminal(&self) -> bool;}

pub struct Transitions {
    first_action: usize,
    first_reward: f32,
    transitions: Vec<(ActionsTaken, usize, f32)>,
    reached_terminal: bool,
}