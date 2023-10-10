use std::collections::BTreeMap;

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
    root_cost: f32,
    root_data: IRStateData,
    data: BTreeMap<ActionsTaken, IRStateData>,
}

pub struct IRStateData {
    frequency: usize,
    actions: Vec<IRActionData>,
}

pub struct IRActionData {
    action: usize,
    frequency: usize,
    probability: f32,
    reward: f32,
    future_reward_sum: f32,
    upper_estimate: f32,
}

impl IRActionData {
    fn new(action: usize, reward: f32, probability: f32) -> Self {
        let q = reward;
        const C_PUCT_0: f32 = 1.0;
        let noise = 1.0;
        let u = q + C_PUCT_0 * probability * noise;
        Self {
            action,
            frequency: 0,
            probability,
            reward,
            future_reward_sum: 0.0,
            upper_estimate: u,
        }
    }

    fn update_future_reward(&mut self, approximate_gain_to_terminal: f32) {
        let Self {
            action: _,
            frequency,
            probability: _,
            reward: _,
            future_reward_sum,
            upper_estimate: _,
        } = self;
        *frequency += 1;
        *future_reward_sum += approximate_gain_to_terminal;
    }

    fn update_upper_estimate(&mut self, frequency: usize) {
        let Self {
            action: _,
            frequency: action_frequency,
            probability,
            reward,
            future_reward_sum,
            upper_estimate,
        } = self;
        let q = *reward + *future_reward_sum / frequency as f32;
        const C_PUCT: f32 = 1.0;
        let noise = (frequency as f32).sqrt() / (1.0 + *action_frequency as f32);
        let u = q + C_PUCT * *probability * noise;
        *upper_estimate = u;
    }
}

impl IRStateData {
    fn new(predicted_probs: &[f32], action_rewards: &[(usize, f32)]) -> Self {
        let actions = action_rewards
            .iter()
            .map(|(a, r)| {
                let p = predicted_probs.get(*a).unwrap();
                IRActionData::new(*a, *r, *p)
            })
            .collect();
        Self {
            frequency: 0,
            actions,
        }
    }

    fn best_action(&self) -> (usize, f32) {
        self.actions.first().map(|a| (a.action, a.reward)).unwrap()
    }

    fn update_future_reward(&mut self, action: usize, approximate_gain_to_terminal: f32) {
        let Self { frequency, actions } = self;
        *frequency += 1;
        let action_data = actions.iter_mut().find(|a| a.action == action).unwrap();
        action_data.update_future_reward(approximate_gain_to_terminal);
        actions
            .iter_mut()
            .for_each(|a| a.update_upper_estimate(*frequency));
        actions.sort_unstable_by(|a, b| b.upper_estimate.total_cmp(&a.upper_estimate));
    }
}

impl<S> IRMinTree<S> {
    pub fn root_cost(&self) -> f32 {
        self.root_cost
    }

    pub fn new(root: &S, probability_predictions: &[f32]) -> Self
    where
        S: Clone + IRState,
    {
        let rewards = root.action_rewards();
        let mut actions: Vec<_> = rewards
            .into_iter()
            .map(|(i, r)| {
                let p = *probability_predictions.get(i).unwrap();
                const C_PUCT: f32 = 1.0;
                let u = r + C_PUCT * p;
                IRActionData {
                    action: i,
                    frequency: 0,
                    probability: p,
                    reward: r,
                    future_reward_sum: 0.0,
                    upper_estimate: u,
                }
            })
            .collect();
        actions.sort_unstable_by(|a, b| b.upper_estimate.total_cmp(&a.upper_estimate));
        let root_data = IRStateData {
            frequency: 0,
            actions,
        };
        Self {
            root_cost: root.cost(),
            root: root.clone(),
            root_data,
            data: Default::default(),
        }
    }

    pub fn simulate_once(&self) -> (Transitions, S)
    where
        S: Clone + IRState,
    {
        let Self {
            root_cost: _,
            root,
            root_data,
            data,
        } = self;
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
                return (
                    Transitions {
                        first_action,
                        first_reward,
                        transitions,
                        new_path: Some(state_path),
                    },
                    state,
                );
            }
        }
        (
            Transitions {
                first_action,
                first_reward,
                transitions,
                new_path: None,
            },
            state,
        )
    }

    pub fn update(&mut self, transitions: &Transitions, gains: &[f32]) {
        let Self {
            root_cost: _,
            root: _,
            root_data,
            data,
        } = self;
        let Transitions {
            first_action,
            first_reward,
            transitions,
            new_path,
        } = transitions;
        /* todo!()
            https://github.com/ariasanovsky/ariasanovsky.github.io/blob/main/content/posts/2023-09-mcts.md
            https://riasanovsky.me/posts/2023-09-mcts/
            currently `approximate_gain_to_terminal` equals g(s) as in this writeup
            we will eventually accommodate g^*(s) and \tilde{g}^*(s)
            the target to optimize is g^*(s)
        */
        assert_eq!(gains.len(), 1);
        let mut approximate_gain_to_terminal = new_path
            .as_ref()
            .map(|_| gains.first().unwrap().max(0.0f32))
            .unwrap_or(0.0f32);
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
            state_data.update_future_reward(*action, approximate_gain_to_terminal);
            approximate_gain_to_terminal += reward;
        });
        approximate_gain_to_terminal += first_reward;
        root_data.update_future_reward(*first_action, approximate_gain_to_terminal);
    }

    pub fn insert(
        &mut self,
        transitions: &Transitions,
        end_state: &S,
        probability_predictions: &[f32],
    ) where
        S: IRState,
    {
        let Self {
            root_cost: _,
            root: _,
            root_data: _,
            data,
        } = self;
        let Transitions {
            first_action: _,
            first_reward: _,
            transitions: _,
            new_path,
        } = transitions;
        if let Some(new_path) = new_path {
            let state_data = IRStateData::new(probability_predictions, &end_state.action_rewards());
            data.insert(new_path.clone(), state_data);
        }
    }

    pub fn observations(&self) -> Vec<f32>
    where
        S: IRState,
    {
        let Self {
            root_cost: _,
            root: _,
            root_data,
            data: _,
        } = self;
        let IRStateData { frequency, actions } = root_data;
        let mut gain_sum = 0.0;
        let mut observations = vec![0.0; S::ACTION + 1];
        actions.iter().for_each(|a| {
            let IRActionData {
                action,
                frequency: action_frequency,
                probability: _,
                reward,
                future_reward_sum,
                upper_estimate: _,
            } = a;
            gain_sum += *reward;
            gain_sum += *future_reward_sum;
            *observations.get_mut(*action).unwrap() = *action_frequency as f32 / *frequency as f32;
        });
        *observations.last_mut().unwrap() = gain_sum / *frequency as f32;
        observations
    }
}

pub trait IRState {
    const ACTION: usize;
    fn cost(&self) -> f32;
    fn action_rewards(&self) -> Vec<(usize, f32)>;
    fn act(&mut self, action: usize);
    fn is_terminal(&self) -> bool;
}

pub struct Transitions {
    first_action: usize,
    first_reward: f32,
    transitions: Vec<(ActionsTaken, usize, f32)>,
    new_path: Option<ActionsTaken>,
}

impl Transitions {
    pub fn costs(&self, root_cost: f32) -> Vec<f32> {
        let mut costs = vec![root_cost];
        let mut cost = root_cost - self.first_reward;
        costs.push(cost);
        let rewards = self.transitions.iter().map(|(_, _, r)| {
            cost -= r;
            cost
        });
        costs.extend(rewards);
        costs
    }
}
