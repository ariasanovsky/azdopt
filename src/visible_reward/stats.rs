pub trait SortedActions {
    type R;
    type G;
    fn best_action(&self) -> (usize, Self::R);
    fn update_future_reward(&mut self, action: usize, reward: &Self::R);
    fn update_futured_reward_and_expected_gain(&mut self, action: usize, reward: &Self::R, gain: &Self::G);
}

pub struct VRewardRootData<R, G> {
    cost: R,
    frequency: usize,
    actions: Vec<VRewardActionData<R, G>>,
}

impl<R, G> VRewardRootData<R, G> {
    pub fn new(cost: R, actions: Vec<(usize, R, f32)>) -> Self
    where
        R: super::Reward,
        G: super::ExpectedFutureGain,
    {
        Self {
            cost,
            frequency: 0,
            actions: actions.into_iter().map(|(action, reward, probability)| VRewardActionData {
                action,
                frequency: 0,
                probability,
                reward,
                future_reward_sum: R::zero(),
                future_gain_estimate_sum: G::zero(),
            }).collect()
        }
    }
}

pub struct VRewardStateData<R, G> {
    frequency: usize,
    actions: Vec<VRewardActionData<R, G>>,
}

pub struct VRewardActionData<R, G> {
    action: usize,
    frequency: usize,
    probability: f32,
    reward: R,
    future_reward_sum: R,
    future_gain_estimate_sum: G,
}

impl<R, G> SortedActions for VRewardRootData<R, G> {
    type R = R;
    type G = G;

    fn best_action(&self) -> (usize, Self::R) {
        todo!()
    }

    fn update_future_reward(&mut self, action: usize, reward: &Self::R) {
        todo!()
    }

    fn update_futured_reward_and_expected_gain(&mut self, action: usize, reward: &Self::R, gain: &Self::G) {
        todo!()
    }
}

impl<R, G> SortedActions for VRewardStateData<R, G> {
    type R = R;
    type G = G;

    fn best_action(&self) -> (usize, Self::R) {
        todo!()
    }

    fn update_future_reward(&mut self, action: usize, reward: &Self::R) {
        todo!()
    }

    fn update_futured_reward_and_expected_gain(&mut self, action: usize, reward: &Self::R, gain: &Self::G) {
        todo!()
    }
}

#[macro_export]
macro_rules! VRewardRootData {
    ($config:ty) => {
        $crate::visible_reward::stats::VRewardRootData<
            <$config as $crate::visible_reward::config::Config>::Reward,
            <$config as $crate::visible_reward::config::Config>::ExpectedFutureGain,
        >
    };
}

#[macro_export]
macro_rules! VRewardStateData {
    ($config:ty) => {
        $crate::visible_reward::stats::VRewardStateData<
            <$config as $crate::visible_reward::config::Config>::Reward,
            <$config as $crate::visible_reward::config::Config>::ExpectedFutureGain,
        >
    };
}
