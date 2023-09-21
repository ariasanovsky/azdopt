pub trait SortedActions {
    type R;
    type G;
    fn best_action(&self) -> (usize, Self::R) where Self::R: Clone;
    fn sort_actions(&mut self);
    fn update_future_reward(&mut self, action: usize, reward: &Self::R)
    where
        Self::R: for<'a> std::ops::AddAssign<&'a Self::R>,
        Self::G: for<'a> std::ops::AddAssign<&'a Self::G>,
    ;
    fn update_futured_reward_and_expected_gain(&mut self, action: usize, reward: &Self::R, gain: &Self::G)
    where
        Self::R: for<'a> std::ops::AddAssign<&'a Self::R>,
        Self::G: for<'a> std::ops::AddAssign<&'a Self::G>,
    ;
}

#[derive(Debug)]
pub struct VRewardRootData<R, G> {
    cost: R,
    frequency: usize,
    actions: Vec<(VRewardActionData<R, G>, f32)>,
}

impl<R, G> VRewardRootData<R, G> {
    pub fn new<C>(cost: R, actions: Vec<(usize, R, f32)>) -> Self
    where
        R: super::Reward,
        G: super::ExpectedFutureGain,
        C: UpperEstimate<VRewardActionData<R, G>>,
    {
        Self {
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
                let upper_estimate = C::upper_estimate(&action_data);
                (action_data, upper_estimate)
            }).collect()
        }
    }
}

#[derive(Debug)]
pub struct VRewardStateData<R, G> {
    frequency: usize,
    actions: Vec<(VRewardActionData<R, G>, f32)>,
}

impl<R, G> VRewardStateData<R, G> {
    pub fn new<C>(actions: Vec<(usize, R, f32)>) -> Self
    where
        R: super::Reward,
        G: super::ExpectedFutureGain,
        C: UpperEstimate<VRewardActionData<R, G>>,
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
                let upper_estimate = C::upper_estimate(&action_data);
                (action_data, upper_estimate)
            }).collect()
        };
        state_data.sort_actions();
        state_data
    }
}

pub trait UpperEstimate<D> {
    fn upper_estimate(data: &D) -> f32;
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
    {
        let (VRewardActionData {
            action: _,
            frequency,
            probability: _,
            reward: _,
            future_reward_sum,
            future_gain_estimate_sum: _,
        }, _) = self.actions.iter_mut().find(|action_data| action_data.0.action == action).unwrap();
        *frequency += 1;
        *future_reward_sum += reward;
        self.sort_actions();
    }

    fn update_futured_reward_and_expected_gain(&mut self, action: usize, reward: &Self::R, gain: &Self::G)
    where
        Self::R: for<'a> std::ops::AddAssign<&'a Self::R>,
        Self::G: for<'a> std::ops::AddAssign<&'a Self::G>,
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

    fn best_action(&self) -> (usize, Self::R) {
        todo!()
    }

    fn sort_actions(&mut self) {
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
