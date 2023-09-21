pub trait SortedActions {
    type R;
    type G;
    type A;
    fn best_action(&self) -> (usize, Self::R) where Self::R: Clone;
    fn sort_actions(&mut self);
    fn update_future_reward(&mut self, action: usize, reward: &Self::R)
    where
        Self::R: for<'a> std::ops::AddAssign<&'a Self::R>,
        Self::G: for<'a> std::ops::AddAssign<&'a Self::G>,
        Self::A: UpperEstimate,
    ;
    fn update_futured_reward_and_expected_gain(&mut self, action: usize, reward: &Self::R, gain: &Self::G)
    where
        Self::R: for<'a> std::ops::AddAssign<&'a Self::R>,
        Self::G: for<'a> std::ops::AddAssign<&'a Self::G>,
        Self::A: UpperEstimate,
    ;
}

pub trait UpperEstimate {
    fn upper_estimate(&self, frequency: usize) -> f32;
}

#[derive(Debug)]
pub struct VRewardRootData<R, G> {
    cost: R,
    frequency: usize,
    actions: Vec<(VRewardActionData<R, G>, f32)>,
}

impl<R, G> VRewardRootData<R, G> {
    pub fn new(cost: R, actions: Vec<(usize, R, f32)>) -> Self
    where
        R: super::Reward,
        G: super::ExpectedFutureGain,
        VRewardActionData<R, G>: UpperEstimate,
    {
        let mut root_data = Self {
            cost,
            frequency: 0,
            actions: actions.into_iter().map(|(action, reward, probability)| {
                let action_data = VRewardActionData {
                    action,
                    frequency: 0,
                    probability,
                    reward,
                    future_reward_sum: R::zero(),
                    future_gain_estimate_sum: G::zero(),
                };
                let upper_estimate = action_data.upper_estimate(0);
                (action_data, upper_estimate)
            }).collect()
        };
        root_data.sort_actions();
        root_data
    }
}

#[derive(Debug)]
pub struct VRewardStateData<R, G> {
    frequency: usize,
    actions: Vec<(VRewardActionData<R, G>, f32)>,
}

impl<R, G> VRewardStateData<R, G> {
    pub fn new(actions: Vec<(usize, R, f32)>) -> Self
    where
        R: super::Reward,
        G: super::ExpectedFutureGain,
        VRewardActionData<R, G>: UpperEstimate,
    {
        let mut state_data = Self {
            frequency: 0,
            actions: actions.into_iter().map(|(action, reward, probability)| {
                let action_data = VRewardActionData {
                    action,
                    frequency: 0,
                    probability,
                    reward,
                    future_reward_sum: R::zero(),
                    future_gain_estimate_sum: G::zero(),
                };
                let upper_estimate = action_data.upper_estimate(0);
                (action_data, upper_estimate)
            }).collect()
        };
        state_data.sort_actions();
        state_data
    }
}

#[derive(Debug)]
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
    type A = VRewardActionData<R, G>;

    fn best_action(&self) -> (usize, Self::R)
    where
        R: Clone,
    {
        self.actions.first().map(|action| (action.0.action, action.0.reward.clone())).unwrap()
    }

    fn sort_actions(&mut self) {
        self.actions.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap()
        });
    }

    fn update_future_reward(&mut self, action: usize, reward: &Self::R)
    where
        Self::R: for<'a> std::ops::AddAssign<&'a Self::R>,
        Self::G: for<'a> std::ops::AddAssign<&'a Self::G>,
        Self::A: UpperEstimate,
    {
        let (action_data, upper_estimate) = self.actions.iter_mut().find(|action_data| action_data.0.action == action).unwrap();
        action_data.frequency += 1;
        action_data.future_reward_sum += reward;
        *upper_estimate = action_data.upper_estimate(self.frequency);
        self.sort_actions();
    }

    fn update_futured_reward_and_expected_gain(&mut self, action: usize, reward: &Self::R, gain: &Self::G)
    where
        Self::R: for<'a> std::ops::AddAssign<&'a Self::R>,
        Self::G: for<'a> std::ops::AddAssign<&'a Self::G>,
        Self::A: UpperEstimate,
    {
        let (VRewardActionData {
            action: _,
            frequency,
            probability: _,
            reward: _,
            future_reward_sum,
            future_gain_estimate_sum,
        }, _) = self.actions.iter_mut().find(|action_data| action_data.0.action == action).unwrap();
        *frequency += 1;
        *future_reward_sum += reward;
        *future_gain_estimate_sum += gain;
        self.sort_actions();
        
    }
}

impl<R, G> SortedActions for VRewardStateData<R, G> {
    type R = R;
    type G = G;
    type A = VRewardActionData<R, G>;

    fn best_action(&self) -> (usize, Self::R) {
        todo!()
    }

    fn sort_actions(&mut self) {
        self.actions.sort_unstable_by(|a, b| {
            b.1.partial_cmp(&a.1).unwrap()
        });
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

impl UpperEstimate for VRewardActionData<i32, f32> {
    fn upper_estimate(&self, frequency: usize) -> f32 {
        let VRewardActionData {
            action: _,
            frequency: action_frequency,
            probability,
            reward,
            future_reward_sum,
            future_gain_estimate_sum,
        } = self;
        let q_prime = if *action_frequency == 0 {
            0.0
        } else {
            (*future_gain_estimate_sum + *future_reward_sum as f32) / *action_frequency as f32
        };
        const C_PUCT: f32 = 1.0;
        let numerator = frequency + 1;
        let denominator = *action_frequency + 1;
        let noise = C_PUCT * *probability * (numerator as f32).sqrt() / denominator as f32;
        *reward as f32 + q_prime + noise
    }
}