use crate::space::StateActionSpace;

use self::action_data::{INTUnvisitedActionData, INTVisitedActionData};

use super::transition::{INTTransition, StateDataKindMutRef, TransitionKind};

pub(crate) mod action_data;

#[derive(Clone, Debug)]
pub enum StateDataKind {
    Exhausted { c_t: f32 },
    Active { data: INTStateData },
}

impl StateDataKind {
    pub fn new<Space>(space: &Space, pi_0_theta: &[f32], c_0: f32, s_0: &Space::State) -> Self
    where
        Space: StateActionSpace,
    {
        if space.is_terminal(s_0) {
            return Self::Exhausted { c_t: c_0 };
        }
        let mut c = 0;
        let p_sum = space
            .action_indices(s_0)
            .map(|a| {
                c += 1;
                pi_0_theta[a]
            })
            .sum::<f32>();
        debug_assert_ne!(c, 0);
        let mut unvisited_actions = space
            .action_indices(s_0)
            .map(|a| INTUnvisitedActionData::new(a, pi_0_theta[a] / p_sum))
            .collect::<Vec<_>>();
        unvisited_actions.sort_by(|a, b| a.p_sa().partial_cmp(&b.p_sa()).unwrap());
        let data = INTStateData {
            n_s: 0,
            c_s: c_0,
            c_s_star: None,
            visited_actions: vec![],
            unvisited_actions,
        };
        Self::Active { data }
        // Self::Active {
        //     n_s: 0,
        //     c_star: cost,
        //     visited_actions: vec![],
        //     unvisited_actions,
        // }
    }

    pub fn cost(&self) -> f32 {
        match self {
            Self::Exhausted { c_t } => *c_t,
            Self::Active { data } => data.c_s,
        }
    }
}

#[derive(Debug, PartialEq, Clone)]
pub struct INTStateData {
    pub(crate) n_s: usize,
    pub(crate) c_s: f32,
    pub(crate) c_s_star: Option<f32>, // todo! math: Option<f32>?
    pub(crate) visited_actions: Vec<INTVisitedActionData>,
    pub(crate) unvisited_actions: Vec<INTUnvisitedActionData>,
}

pub struct UpperEstimateData {
    pub n_s: usize,
    pub n_sa: usize,
    pub g_sa_sum: f32,
    pub p_sa: f32,
    pub depth: usize,
}

impl INTStateData {
    pub fn len(&self) -> usize {
        self.visited_actions.len() + self.unvisited_actions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.visited_actions.is_empty() && self.unvisited_actions.is_empty()
    }

    pub fn write_observations(&self, probs: &mut [f32], values: &mut [f32]) {
        probs.fill(0.0);
        debug_assert_eq!(values.len(), 1);
        let Self {
            n_s,
            c_s: _,
            c_s_star: _,
            visited_actions,
            unvisited_actions: _,
        } = self;
        // todo!();
        let n_s = *n_s as f32;
        let mut value = 0.0;
        let n_sum = visited_actions
            .iter()
            // .filter(|a| a.n_sa() != 0)
            .map(|a| {
                debug_assert_ne!(a.n_sa(), 0);
                let n_sa = a.n_sa() as f32;
                let g_sa = a.g_sa();
                probs[a.action()] = n_sa / n_s;
                value += g_sa;
                n_sa
            })
            .sum::<f32>();
        debug_assert_eq!(n_sum, n_s);
        values[0] = value / n_s;
    }

    // require 2 separate upper estimate functions?
    pub fn best_action(
        &mut self,
        upper_estimate: impl Fn(UpperEstimateData) -> f32,
    ) -> Option<INTTransition> {
        let Self {
            n_s: _,
            c_s: _,
            c_s_star: _,
            visited_actions,
            unvisited_actions,
        } = self;
        visited_actions.iter_mut().for_each(|a| {
            let est_data = a.upper_estimate_data(self.n_s, 0);
            a.set_upper_estimate(upper_estimate(est_data));
        });
        visited_actions.sort_by(|a, b| {
            let u_sa = upper_estimate(a.upper_estimate_data(self.n_s, 0));
            let u_sb = upper_estimate(b.upper_estimate_data(self.n_s, 0));
            u_sa.partial_cmp(&u_sb).unwrap()
        });
        let best_visited_action_estimate: Option<f32> =
            visited_actions.last().map(|a| a.upper_estimate());
        let best_unvisited_action_estimate: Option<f32> = unvisited_actions.last().map(|a| {
            let est_data = a.upper_estimate_data(self.n_s, 0);
            upper_estimate(est_data)
        });
        // dbg!(best_visited_action_estimate, best_unvisited_action_estimate);
        let kind = match (best_visited_action_estimate, best_unvisited_action_estimate) {
            (None, None) => None,
            (None, Some(_)) => Some(TransitionKind::LastUnvisitedAction),
            (Some(_), None) => Some(TransitionKind::LastVisitedAction),
            (Some(v), Some(u)) => Some(match v.partial_cmp(&u).unwrap() {
                std::cmp::Ordering::Less => TransitionKind::LastUnvisitedAction,
                std::cmp::Ordering::Equal | std::cmp::Ordering::Greater => {
                    TransitionKind::LastVisitedAction
                }
            }),
        };
        kind.map(|kind| INTTransition {
            data_i: StateDataKindMutRef::Active { data: self },
            kind,
        })
    }
}
