use crate::int_min_tree::state_data::action_data::{INTUnvisitedActionData, INTVisitedActionData};

use super::state_data::INTStateData;

#[derive(Debug)]
pub struct INTTransition<'a> {
    pub(crate) data_i: StateDataKindMutRef<'a>,
    pub(crate) kind: TransitionKind,
}

#[derive(Debug)]
pub enum StateDataKindMutRef<'a> {
    Exhausted { c_t_star: f32 },
    Active { data: &'a mut INTStateData },
}

impl<'a> INTTransition<'a> {
    pub fn index(&self) -> usize {
        match self {
            Self {
                data_i: StateDataKindMutRef::Exhausted { c_t_star: _ },
                kind: _,
            } => unreachable!("Exhausted state has no best action"),
            Self {
                data_i: StateDataKindMutRef::Active { data },
                kind,
            } => match kind {
                TransitionKind::LastUnvisitedAction => data.unvisited_actions.last().unwrap().a,
                TransitionKind::LastVisitedAction => data.visited_actions.last().unwrap().a,
            },
        }
    }

    pub(crate) fn cascade_update(&mut self, c_star_i_plus_one: &mut f32) -> bool {
        todo!("when used in `update_existing_nodes`, we seem to remove all actions from a state but do not mark it exhausted; to investigate");
        dbg!();
        let Self { data_i, kind } = self;
        let exhausted = match data_i {
            StateDataKindMutRef::Exhausted { c_t_star: _ } => {
                unreachable!("cascade update only applies to exhausted states")
            }
            StateDataKindMutRef::Active { data } => {
                let INTStateData {
                    n_s,
                    c_s,
                    c_s_star,
                    visited_actions,
                    unvisited_actions,
                } = data;
                *n_s += 1;
                // *c_star_i_plus_one = c_star_i_plus_one.min(*c_s);
                *c_star_i_plus_one = c_star_i_plus_one.min(*c_s_star);
                *c_s_star = *c_star_i_plus_one;
                match kind {
                    TransitionKind::LastUnvisitedAction => {
                        // remove the last unvisited action from `data`
                        let crate::int_min_tree::state_data::action_data::INTUnvisitedActionData {
                            a: _,
                            p_sa: _,
                        } = unvisited_actions.pop().expect("no unvisited actions");
                    }
                    TransitionKind::LastVisitedAction => {
                        let crate::int_min_tree::state_data::action_data::INTVisitedActionData {
                            a: _,
                            p_sa: _,
                            n_sa: _,
                            g_sa_sum: _,
                            u_sa: _,
                        } = visited_actions.pop().expect("no visited actions");
                    }
                }
                visited_actions.is_empty() && unvisited_actions.is_empty()
            }
        };
        // we only sort during `best_action` calls
        if exhausted {
            *data_i = StateDataKindMutRef::Exhausted {
                c_t_star: *c_star_i_plus_one,
            };
        }
        exhausted
    }

    pub(crate) fn update(&mut self, c_star_theta_i_plus_one: &mut f32) {
        let Self { data_i, kind } = self;
        match data_i {
            StateDataKindMutRef::Exhausted { c_t_star: _ } => {
                unreachable!("updating here implies we took an action from an exhausted state")
            }
            StateDataKindMutRef::Active { data } => {
                let INTStateData {
                    n_s,
                    c_s,
                    c_s_star,
                    visited_actions,
                    unvisited_actions,
                } = data;
                *n_s += 1;
                match kind {
                    TransitionKind::LastUnvisitedAction => {
                        // remove the last unvisited action from `data` to move it to `visited_actions`
                        let INTUnvisitedActionData { a, p_sa } =
                            unvisited_actions.pop().expect("no unvisited actions");
                        let visited_action_data = INTVisitedActionData {
                            a,
                            p_sa,
                            n_sa: 1,
                            g_sa_sum: *c_s_star - *c_star_theta_i_plus_one,
                            // g_sa_sum: *c_s - *c_star_theta_i_plus_one,
                            u_sa: 0.0,
                        };
                        // `u_sa` in an invalid state, but we'll update it before sorting
                        visited_actions.push(visited_action_data);
                    }
                    TransitionKind::LastVisitedAction => {
                        let visited_action =
                            visited_actions.last_mut().expect("no visited actions");
                        visited_action.update(*c_s - *c_star_theta_i_plus_one);
                        // `u_sa` in an invalid state, but we'll update it before sorting
                    }
                }
                // *c_star_theta_i_plus_one = c_star_theta_i_plus_one.min(*c_s);
                *c_star_theta_i_plus_one = c_star_theta_i_plus_one.min(*c_s_star);
                *c_s_star = *c_star_theta_i_plus_one;
            }
        }
    }
}

#[derive(Debug)]
pub enum TransitionKind {
    LastUnvisitedAction,
    LastVisitedAction,
}
