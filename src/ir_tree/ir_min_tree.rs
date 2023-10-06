use std::collections::BTreeMap;

use super::stats::{VRewardRootData, VRewardStateData, VRewardActionData};

pub struct ActionsTaken {
    actions_taken: Vec<usize>,
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
}

pub trait IRState {
    fn cost(&self) -> f32;
    fn action_rewards(&self) -> Vec<(usize, f32)>;
}