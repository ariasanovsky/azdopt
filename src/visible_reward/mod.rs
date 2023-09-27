use std::collections::BTreeMap;

pub mod config;
pub mod log;
pub mod model;
pub mod stats;
pub mod transitions;

use config::*;

use self::{
    log::{FinalStateData, Log},
    stats::{SortedActions, UpperEstimate, Observation},
    transitions::{FinalState, Transitions}, model::Model,
};

pub struct VRewardTree<S, P, D0, D> {
    pub root: S,
    pub root_data: D0,
    pub data: BTreeMap<P, D>,
}

impl<S, P, D0, D> VRewardTree<S, P, D0, D> {
    pub fn new<C>(root: S, root_prediction: C::P) -> Self
    where
        C: HasPrediction,
        C: HasReward,
        C::P: Prediction<D, D0, R = C::R>,
        S: State<R = C::R>,
    {
        let root_cost = root.cost();
        let transitions = root.action_rewards();
        let root_data = root_prediction.new_root_data(root_cost, transitions);
        Self {
            root,
            root_data,
            data: BTreeMap::new(),
        }
    }

    pub fn simulate_once<C>(&self) -> Transitions<C::R, P, S>
    where
        S: Clone + State,
        D0: SortedActions<R = C::R>,
        C: HasReward,
        C::R: Clone,
        P: Path + Clone + Ord,
        D: SortedActions<R = C::R>,
    {
        let mut state = self.root.clone();
        let (first_action, first_reward) = self.root_data.best_action();
        state.act(first_action);
        let mut state_path = P::new(first_action);
        let mut transitions = vec![];
        while !state.is_terminal() {
            if let Some(data) = self.data.get(&state_path) {
                let (action, reward) = data.best_action();
                state.act(action);
                transitions.push((state_path.clone(), action, reward));
                state_path.push(action);
            } else {
                // new state
                return Transitions {
                    a1: first_action,
                    r1: first_reward,
                    transitions,
                    end: FinalState::New(state_path, state),
                };
            }
        }
        // terminal state
        Transitions {
            a1: first_action,
            r1: first_reward,
            transitions,
            end: FinalState::Leaf(state_path, state),
        }
    }

    pub fn insert<C>(&mut self, path: P, rewards: Vec<(usize, C::R)>, prediction: C::P) -> C::G
    where
        C: HasReward,
        C: HasPrediction,
        C: HasExpectedFutureGain,
        C::P: Prediction<D, D0, R = C::R, G = C::G>,
        P: Ord,
    {
        let (data, gain) = prediction.new_data(rewards);
        self.data.insert(path, data);
        gain
    }

    pub fn update_with_transitions<C>(
        &mut self,
        first_action: usize,
        transitions: Vec<(P, usize, C::R)>,
    ) where
        C: HasReward,
        C::R: Reward,
        C::R: for<'a> core::ops::AddAssign<&'a C::R>,
        C: HasExpectedFutureGain,
        C::G: ExpectedFutureGain,
        C::G: for<'a> core::ops::AddAssign<&'a C::G>,
        P: Ord,
        D: SortedActions<R = C::R, G = C::G>,
        D0: SortedActions<R = C::R, G = C::G>,
        D::A: UpperEstimate,
        D0::A: UpperEstimate,
    {
        /* we have the vector (p_1, a_2, r_2), ..., (p_{t-1}, a_t, r_t)
            we need to update p_{t-1} (s_{t-1}) with n(s_{t-1}, a_t) += 1 & n(s_{t-1}) += 1
            ...
            then p_i (s_i) with the future reward r_{i+2} + ... + r_t as well as the n increments
            ...
            then p_1 (s_1) with the future reward r_3 + ... + r_t as well as the n increments
            then p_0 (s_0) with the future reward r_2 + ... + r_t as well as the n increments
        */
        let mut reward_sum = C::R::zero();
        transitions.iter().rev().for_each(|(path, action, reward)| {
            let data = self.data.get_mut(path).unwrap();
            data.update_future_reward(*action, &reward_sum);
            reward_sum += reward;
        });
        self.root_data
            .update_future_reward(first_action, &reward_sum);
    }

    pub fn update_with_transitions_and_evaluation<C>(
        &mut self,
        first_action: usize,
        transitions: Vec<(P, usize, C::R)>,
        evaluation: C::G,
    ) where
        C: HasReward,
        C::R: Reward,
        C::R: for<'a> core::ops::AddAssign<&'a C::R>,
        C: HasExpectedFutureGain,
        C::G: ExpectedFutureGain,
        C::G: for<'a> core::ops::AddAssign<&'a C::G>,
        P: Ord,
        D: SortedActions<R = C::R, G = C::G>,
        D0: SortedActions<R = C::R, G = C::G>,
        D::A: UpperEstimate,
        D0::A: UpperEstimate,
    {
        /* we have the vector (p_1, a_2, r_2), ..., (p_{t-1}, a_t, r_t)
            we need to update p_{t-1} (s_{t-1}) with n(s_{t-1}, a_t) += 1 & n(s_{t-1}) += 1
            we also need to update p_{t-1} with the expected future gain g
            ...
            then p_i (s_i) with the future reward r_{i+2} + ... + r_t + g as well as the n increments
            ...
            then p_1 (s_1) with the future reward r_3 + ... + r_t + g as well as the n increments
            then p_0 (s_0) with the future reward r_2 + ... + r_t + g as well as the n increments
        */
        let mut reward_sum = C::R::zero();
        transitions.iter().rev().for_each(|(path, action, reward)| {
            let data = self.data.get_mut(path).unwrap();
            data.update_futured_reward_and_expected_gain(*action, &reward_sum, &evaluation);
            reward_sum += reward;
        });
        self.root_data.update_futured_reward_and_expected_gain(
            first_action,
            &reward_sum,
            &evaluation,
        );
    }

    pub fn simulate_once_and_update<C>(&mut self, model: &C::M, log: &mut C::L)
    where
        C: HasReward,
        C::R: Clone,
        S: Clone + State<R = C::R>,
        D0: SortedActions<R = C::R>,
        P: Path + Clone + Ord,
        D: SortedActions<R = C::R>,
        C: HasReward,
        C::R: Reward,
        C::R: for<'a> core::ops::AddAssign<&'a C::R>,
        C: HasExpectedFutureGain,
        C::G: ExpectedFutureGain + Clone,
        C::G: for<'a> core::ops::AddAssign<&'a C::G>,
        P: Ord,
        D: SortedActions<R = C::R, G = C::G>,
        D0: SortedActions<R = C::R, G = C::G>,
        D::A: UpperEstimate,
        D0::A: UpperEstimate,
        C: HasModel,
        C: HasPrediction,
        C::P: Prediction<D, D0, R = C::R, G = C::G>,
        C::M: Model<S, P = C::P>,
        C: HasLog + HasEndNode<E = FinalStateData<C::G>>,
        C::L: Log<R = C::R, T = Vec<(P, usize, C::R)>, G = C::G>,
    {
        let transitions = self.simulate_once::<C>();
        let Transitions {
            a1,
            r1,
            transitions,
            end,
        } = transitions;
        match end {
            FinalState::Leaf(_, _) => {
                // let trans = transitions.iter().map(|(_, a, r)| (a, r));
                log.add_transition_data(a1, r1, &transitions, FinalStateData::Leaf);
                self.update_with_transitions::<C>(a1, transitions);
            }
            FinalState::New(p, s) => {
                let prediction = model.predict(&s);
                log.add_transition_data(
                    a1,
                    r1,
                    &transitions,
                    FinalStateData::New(prediction.value().clone()),
                );
                let t = s.action_rewards();
                let eval = self.insert::<C>(p, t, prediction);
                self.update_with_transitions_and_evaluation::<C>(a1, transitions, eval)
            }
        }
    }

    pub fn to_observation(self) -> (S, (Vec<(usize, f32)>, f32))
    where
        D0: Observation<O = (Vec<(usize, f32)>, f32)>,
        D0::R: Reward + core::ops::AddAssign<D0::R>,
        D0::G: ExpectedFutureGain + core::ops::AddAssign<D0::G>,
    {
        let Self { root, root_data, data: _ } = self;
        let obs = root_data.to_observation();
        (root, obs)
    }
}

pub trait Reward {
    fn zero() -> Self;
    fn to_f32(&self) -> f32;
}

impl Reward for i32 {
    fn zero() -> Self {
        0
    }
    fn to_f32(&self) -> f32 {
        *self as f32
    }
}

pub trait Prediction<D, D0> {
    type G;
    type R;
    fn value(&self) -> &Self::G;
    fn new_data(&self, transitions: Vec<(usize, Self::R)>) -> (D, Self::G);
    fn new_root_data(&self, cost: Self::R, transitions: Vec<(usize, Self::R)>) -> D0;
}

pub trait ExpectedFutureGain {
    fn zero() -> Self;
    fn to_f32(&self) -> f32;
}

impl ExpectedFutureGain for f32 {
    fn zero() -> Self {
        0.0
    }
    fn to_f32(&self) -> f32 {
        *self
    }
}

pub trait State {
    type R;
    fn is_terminal(&self) -> bool;
    fn act(&mut self, action: usize);
    fn cost(&self) -> Self::R;
    fn action_rewards(&self) -> Vec<(usize, Self::R)>;
}

pub trait Path {
    fn new(action: usize) -> Self;
    fn push(&mut self, action: usize);
}
