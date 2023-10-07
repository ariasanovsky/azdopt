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

    pub fn update(&mut self, transitions: &Transitions, gain_prediction: &[f32]) {
        let Self {
            root: _,
            root_data,
            data,
        } = self;
        let Transitions {
            first_action,
            first_reward,
            transitions,
            reached_terminal
        } = transitions;
        /* todo!()
            https://github.com/ariasanovsky/ariasanovsky.github.io/blob/main/content/posts/2023-09-mcts.md
            https://riasanovsky.me/posts/2023-09-mcts/
            currently `approximate_gain_to_terminal` equals g(s) as in this writeup
            we will eventually accommodate g^*(s) and \tilde{g}^*(s)
            the target to optimize is g^*(s)
        */
        assert_eq!(gain_prediction.len(), 1);
        let mut approximate_gain_to_terminal = if *reached_terminal {
            0.0f32
        } else {
            0.0f32.max(*gain_prediction.first().unwrap())
        };
        /* we have the vector (p_1, a_2, r_2), ..., (p_{t-1}, a_t, r_t)
            we need to update p_{t-1} (s_{t-1}) with n(s_{t-1}, a_t) += 1 & n(s_{t-1}) += 1
            ...
            then p_i (s_i) with the future reward r_{i+2} + ... + r_t as well as the n increments
            ...
            then p_1 (s_1) with the future reward r_3 + ... + r_t as well as the n increments
            then p_0 (s_0) with the future reward r_2 + ... + r_t as well as the n increments
        */
        transitions.iter().rev().for_each(|(path, action, reward)| {
            let state_data = data.get_mut(path).unwrap();
            state_data.update_future_reward(*action, &approximate_gain_to_terminal);
            approximate_gain_to_terminal += reward;
        });
        root_data.update_future_reward(*first_action, &approximate_gain_to_terminal);
    }

    pub fn observations(&self) -> Vec<f32> {
        let Self {
            root: _,
            root_data,
            data: _,
        } = self;
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