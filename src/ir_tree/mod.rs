use std::collections::BTreeMap;

pub mod config;
pub mod log;
pub mod model;
pub mod stats;
pub mod transitions;

pub mod ir_min_tree;

use config::*;

use self::{
    log::{FinalStateData, Log},
    stats::{SortedActions, UpperEstimate, Observation},
    transitions::{FinalState, Transitions}, model::Model,
};

#[derive(Debug)]
pub struct IRTree<S, P, D0, D> {
    pub root: S,
    pub root_data: D0,
    pub data: BTreeMap<P, D>,
}

impl<S, P, D0, D> IRTree<S, P, D0, D> {
    pub fn new<C>(root: S, root_prediction: C::P) -> Self
    where
        C: HasPrediction,
        C::P: Prediction<D, D0>,
        S: State,
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

    pub fn simulate_once<C>(&self) -> Transitions<P, S>
    where
        S: Clone + State,
        D0: SortedActions,
        P: Path + Clone + Ord,
        D: SortedActions,
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

    pub fn insert<C>(&mut self, path: P, rewards: Vec<(usize, f32)>, prediction: C::P) -> f32
    where
        C: HasPrediction,
        C::P: Prediction<D, D0>,
        P: Ord,
    {
        let (data, gain) = prediction.new_data(rewards);
        self.data.insert(path, data);
        gain
    }

    pub fn update_with_transitions<C>(
        &mut self,
        first_action: usize,
        transitions: Vec<(P, usize, f32)>,
    ) where
        P: Ord,
        D: SortedActions,
        D0: SortedActions,
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
        let mut reward_sum = 0.0;
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
        transitions: Vec<(P, usize, f32)>,
        evaluation: f32,
    ) where
        P: Ord,
        D: SortedActions,
        D0: SortedActions,
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
        let mut reward_sum = 0.0;
        transitions.iter().rev().for_each(|(path, action, reward)| {
            let data = self.data.get_mut(path).unwrap();
            data.update_future_reward(*action, &(reward_sum + evaluation));
            reward_sum += reward;
        });
        self.root_data.update_future_reward(
            first_action,
            &(reward_sum + evaluation),
        );
    }

    pub fn simulate_once_and_update<C>(&mut self, model: &C::M, log: &mut C::L)
    where
        S: Clone + State,
        D0: SortedActions,
        P: Path + Clone + Ord,
        D: SortedActions,
        D: SortedActions,
        D0: SortedActions,
        D::A: UpperEstimate,
        D0::A: UpperEstimate,
        C: HasModel,
        C: HasPrediction,
        C::P: Prediction<D, D0>,
        C::M: Model<S, P = C::P>,
        C: HasLog + HasEndNode<E = FinalStateData>,
        C::L: Log<T = Vec<(P, usize, f32)>>,
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
                    FinalStateData::New { final_reward: prediction.value().clone() },
                );
                let t = s.action_rewards();
                let eval = self.insert::<C>(p, t, prediction);
                self.update_with_transitions_and_evaluation::<C>(a1, transitions, eval)
            }
        }
    }

    pub fn to_observation(self) -> (S, (Vec<(usize, f32)>, f32))
    where
        D0: Observation,
    {
        let Self { root, root_data, data: _ } = self;
        let obs = root_data.to_observation();
        (root, obs)
    }
}

pub trait Prediction<D, D0> {
    fn value(&self) -> &f32;
    fn new_data(&self, transitions: Vec<(usize, f32)>) -> (D, f32);
    fn new_root_data(&self, cost: f32, transitions: Vec<(usize, f32)>) -> D0;
}

pub trait State {
    fn is_terminal(&self) -> bool;
    fn act(&mut self, action: usize);
    fn cost(&self) -> f32;
    fn action_rewards(&self) -> Vec<(usize, f32)>;
}

pub trait Path {
    fn new(action: usize) -> Self;
    fn push(&mut self, action: usize);
}
