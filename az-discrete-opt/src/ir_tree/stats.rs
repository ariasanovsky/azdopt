pub trait SortedActions {
    type A;
    fn best_action(&self) -> (usize, f32);
    fn sort_actions(&mut self);
    fn update_future_reward(&mut self, action: usize, reward: &f32)
    where
        Self::A: UpperEstimate;
}

pub trait UpperEstimate {
    fn upper_estimate(&self, frequency: usize) -> f32;
}

#[derive(Debug)]
pub struct VRewardRootData {
    pub cost: f32,
    pub frequency: usize,
    pub actions: Vec<(VRewardActionData, f32)>,
}

impl VRewardRootData {
    pub fn new(cost: f32, actions: Vec<(usize, f32, f32)>) -> Self {
        let mut root_data = Self {
            cost,
            frequency: 0,
            actions: actions
                .into_iter()
                .map(|(action, reward, probability)| {
                    let action_data = VRewardActionData {
                        action,
                        frequency: 0,
                        probability,
                        reward,
                        future_reward_sum: 0.0,
                    };
                    let upper_estimate = action_data.upper_estimate(0);
                    (action_data, upper_estimate)
                })
                .collect(),
        };
        root_data.sort_actions();
        root_data
    }
}

#[derive(Debug)]
pub struct VRewardStateData {
    pub(crate) frequency: usize,
    pub(crate) actions: Vec<(VRewardActionData, f32)>,
}

impl VRewardStateData {
    pub fn new(actions: Vec<(usize, f32, f32)>) -> Self {
        let mut state_data = Self {
            frequency: 0,
            actions: actions
                .into_iter()
                .map(|(action, reward, probability)| {
                    let action_data = VRewardActionData {
                        action,
                        frequency: 0,
                        probability,
                        reward,
                        future_reward_sum: 0.0,
                    };
                    let upper_estimate = action_data.upper_estimate(0);
                    (action_data, upper_estimate)
                })
                .collect(),
        };
        state_data.sort_actions();
        state_data
    }
}

#[derive(Debug)]
pub struct VRewardActionData {
    pub(crate) action: usize,
    pub(crate) frequency: usize,
    pub(crate) probability: f32,
    pub(crate) reward: f32,
    pub(crate) future_reward_sum: f32,
}

impl SortedActions for VRewardRootData {
    type A = VRewardActionData;

    fn best_action(&self) -> (usize, f32) {
        self.actions
            .first()
            .map(|action| (action.0.action, action.0.reward))
            .unwrap()
    }

    fn sort_actions(&mut self) {
        self.actions
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    }

    fn update_future_reward(&mut self, action: usize, reward: &f32) {
        let Self {
            cost: _,
            frequency: state_frequency,
            actions,
        } = self;
        *state_frequency += 1;
        let (action_data, upper_estimate) = actions
            .iter_mut()
            .find(|action_data| action_data.0.action == action)
            .unwrap();
        let VRewardActionData {
            action: _,
            frequency: action_frequency,
            probability: _,
            reward: _,
            future_reward_sum,
        } = action_data;
        *action_frequency += 1;
        *future_reward_sum += reward;
        *upper_estimate = action_data.upper_estimate(*state_frequency);

        self.sort_actions();
    }
}

impl SortedActions for VRewardStateData {
    type A = VRewardActionData;

    fn best_action(&self) -> (usize, f32) {
        self.actions
            .first()
            .map(|action| (action.0.action, action.0.reward))
            .unwrap()
    }

    fn sort_actions(&mut self) {
        self.actions
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    }

    fn update_future_reward(&mut self, action: usize, reward: &f32) {
        let Self {
            frequency: state_frequency,
            actions: _,
        } = self;
        *state_frequency += 1;
        let (action_data, upper_estimate) = self
            .actions
            .iter_mut()
            .find(|action_data| action_data.0.action == action)
            .unwrap();
        let VRewardActionData {
            action: _,
            frequency: action_frequency,
            probability: _,
            reward: _,
            future_reward_sum,
        } = action_data;
        *action_frequency += 1;
        *future_reward_sum += reward;
        *upper_estimate = action_data.upper_estimate(*state_frequency);
        self.sort_actions();
    }
}

// #[macro_export]
// macro_rules! VRewardRootData {
//     ($config:ty) => {
//         $crate::visible_reward::stats::VRewardRootData<
//             <$config as $crate::visible_reward::config::Config>::Reward,
//             <$config as $crate::visible_reward::config::Config>::ExpectedFutureGain,
//         >
//     };
// }

// #[macro_export]
// macro_rules! VRewardStateData {
//     ($config:ty) => {
//         $crate::visible_reward::stats::VRewardStateData<
//             <$config as $crate::visible_reward::config::Config>::Reward,
//             <$config as $crate::visible_reward::config::Config>::ExpectedFutureGain,
//         >
//     };
// }

impl UpperEstimate for VRewardActionData {
    fn upper_estimate(&self, state_frequency: usize) -> f32 {
        let VRewardActionData {
            action: _,
            frequency: action_frequency,
            probability,
            reward,
            future_reward_sum,
        } = self;
        let q_prime = if *action_frequency == 0 {
            0.0
        } else {
            *future_reward_sum / *action_frequency as f32
        };
        const C_PUCT: f32 = 1.0;
        let numerator = state_frequency + 1;
        let denominator = *action_frequency + 1;
        let noise = C_PUCT * *probability * (numerator as f32).sqrt() / denominator as f32;
        *reward + q_prime + noise
    }
}

pub trait Observation {
    fn to_observation(self) -> (Vec<(usize, f32)>, f32);
}

impl Observation for VRewardRootData {
    fn to_observation(self) -> (Vec<(usize, f32)>, f32) {
        let Self {
            cost: _,
            frequency: state_frequency,
            actions,
        } = self;
        let mut total_rewards = 0.0;
        let probs: Vec<(_, _)> = actions
            .into_iter()
            .map(|(action_data, _upper_estimate)| {
                let VRewardActionData {
                    action,
                    frequency: action_frequency,
                    probability: _,
                    reward: _,
                    future_reward_sum,
                } = action_data;
                total_rewards += future_reward_sum;
                let prob = action_frequency as f32 / state_frequency as f32;
                (action, prob)
            })
            .collect();
        let expected_future_gain = total_rewards / state_frequency as f32;
        (probs, expected_future_gain)
    }
}

// pub trait Float {
//     fn to_f32(&self) -> f32;
// }
