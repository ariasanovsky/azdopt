use crate::space::StateActionSpace;

use self::action_data::{INTUnvisitedActionData, INTVisitedActionData};

use super::transition::{INTTransition, StateDataKindMutRef, TransitionKind};

pub(crate) mod action_data;

#[derive(Debug)]
pub enum StateDataKind {
    Exhausted { c_t_star: f32 },
    Active { data: INTStateData },
}

impl StateDataKind {
    pub fn new<Space>(probs: &[f32], cost: f32, state: &Space::State) -> Self
    where
        Space: StateActionSpace,
    {
        if Space::is_terminal(state) {
            return Self::Exhausted { c_t_star: cost };
        }
        let p_sum = Space::actions(state).map(|a| probs[a]).sum::<f32>();
        let mut unvisited_actions = Space::actions(state)
            .map(|a| INTUnvisitedActionData {
                a,
                p_sa: probs[a] / p_sum,
            })
            .collect::<Vec<_>>();
        unvisited_actions.sort_by(|a, b| a.p_sa.partial_cmp(&b.p_sa).unwrap());
        let data = INTStateData {
            n_s: 0,
            c_star: cost,
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
}

#[derive(Debug, PartialEq)]
pub struct INTStateData {
    pub(crate) n_s: usize,
    pub(crate) c_star: f32,
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
    pub fn observe(&self, probs: &mut [f32], values: &mut [f32]) {
        probs.fill(0.0);
        debug_assert_eq!(values.len(), 1);
        let Self {
            n_s,
            c_star: _,
            visited_actions,
            unvisited_actions: _,
        } = self;
        // todo!();
        let n_s = *n_s as f32;
        let mut value = 0.0;
        let n_sum = visited_actions
            .iter()
            .filter(|a| a.n_sa != 0)
            .map(|a| {
                let n_sa = a.n_sa as f32;
                let q_sa = a.g_sa_sum / n_sa;
                probs[a.a] = n_sa / n_s;
                value += q_sa;
                n_sa
            })
            .sum::<f32>();
        debug_assert_eq!(n_sum, n_s);
        values[0] = value / n_s;
    }

    pub fn best_action(
        &mut self,
        upper_estimate: &impl Fn(UpperEstimateData) -> f32,
    ) -> Option<INTTransition> {
        let Self {
            n_s: _,
            c_star: _,
            visited_actions,
            unvisited_actions,
        } = self;
        // let upper_estimate = |g_sa_sum: f32, p_sa: f32, n_s: usize, n_sa: usize| {
        //     debug_assert_ne!(n_s, 0);
        //     debug_assert_ne!(n_sa, 0);
        //     let n_s = n_s as f32;
        //     let n_sa = n_sa as f32;
        //     let p_sa = p_sa;
        //     let c_puct = C_PUCT;
        //     let g_sa = g_sa_sum / n_sa;
        //     let u_sa = g_sa + c_puct * p_sa * (n_s.sqrt() / n_sa);
        //     // println!(
        //     //     "{u_sa} = {g_sa_sum} / {n_sa} + {c_puct} * {p_sa} * ({n_s}.sqrt() / {n_sa})",
        //     // );
        //     u_sa
        // };
        visited_actions.iter_mut().for_each(|a| {
            let INTVisitedActionData {
                a: _,
                p_sa,
                n_sa,
                g_sa_sum,
                u_sa,
            } = a;
            let est_data = UpperEstimateData {
                n_s: self.n_s,
                n_sa: *n_sa,
                g_sa_sum: *g_sa_sum,
                p_sa: *p_sa,
                depth: 0,
            };
            *u_sa = upper_estimate(est_data);
        });
        visited_actions.sort_by(|a, b| a.u_sa.partial_cmp(&b.u_sa).unwrap());
        let best_visited_action_estimate: Option<f32> = visited_actions.last().map(|a| a.u_sa);
        let best_unvisited_action_estimate: Option<f32> = unvisited_actions.last().map(|a| {
            let INTUnvisitedActionData { a: _, p_sa } = a;
            let est_data = UpperEstimateData {
                n_s: self.n_s,
                n_sa: 1,
                g_sa_sum: 0.0,
                p_sa: *p_sa,
                depth: 0,
            };
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
