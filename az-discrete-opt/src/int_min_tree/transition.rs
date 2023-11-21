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
                TransitionKind::LastUnvisitedAction => data.unvisited_actions.last().unwrap().action(),
                TransitionKind::LastVisitedAction => data.visited_actions.last().unwrap().action(),
            },
        }
    }

    pub(crate) fn cascade_update(&mut self, c_star_i_plus_one: &mut f32) -> bool {
        todo!("when used in `update_existing_nodes`, we seem to remove all actions from a state but do not mark it exhausted; to investigate");
        dbg!();
        // let Self { data_i, kind } = self;
        // let exhausted = match data_i {
        //     StateDataKindMutRef::Exhausted { c_t_star: _ } => {
        //         unreachable!("cascade update only applies to exhausted states")
        //     }
        //     StateDataKindMutRef::Active { data } => {
        //         let INTStateData {
        //             n_s,
        //             c_s,
        //             c_s_star,
        //             visited_actions,
        //             unvisited_actions,
        //         } = data;
        //         *n_s += 1;
        //         // *c_star_i_plus_one = c_star_i_plus_one.min(*c_s);
        //         *c_star_i_plus_one = c_star_i_plus_one.min(*c_s_star);
        //         *c_s_star = *c_star_i_plus_one;
        //         match kind {
        //             TransitionKind::LastUnvisitedAction => {
        //                 // remove the last unvisited action from `data`
        //                 let _ = unvisited_actions.pop().expect("no unvisited actions");
        //             }
        //             TransitionKind::LastVisitedAction => {
        //                 let _ = visited_actions.pop().expect("no visited actions");
        //             }
        //         }
        //         visited_actions.is_empty() && unvisited_actions.is_empty()
        //     }
        // };
        // // we only sort during `best_action` calls
        // if exhausted {
        //     *data_i = StateDataKindMutRef::Exhausted {
        //         c_t_star: *c_star_i_plus_one,
        //     };
        // }
        // exhausted
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
                // todo! this performs better with:
                let g_star_theta_i = c_s_star.unwrap_or(*c_s) - *c_star_theta_i_plus_one;
                // let g_star_theta_i = *c_s - *c_star_theta_i_plus_one;
                match kind {
                    TransitionKind::LastUnvisitedAction => {
                        // remove the last unvisited action from `data` to move it to `visited_actions`
                        let unvisited_data = unvisited_actions.pop().expect("no unvisited actions");
                        let visited_data =
                            unvisited_data.to_visited_action(g_star_theta_i);
                        visited_actions.push(visited_data);
                    }
                    TransitionKind::LastVisitedAction => {
                        let visited_action =
                            visited_actions.last_mut().expect("no visited actions");
                        visited_action.update(g_star_theta_i);
                    }
                }
                // todo! math
                match c_s_star {
                    Some(c_s_star) => {
                        *c_s_star = c_s_star.min(*c_star_theta_i_plus_one);
                    }
                    None => *c_s_star = Some(*c_star_theta_i_plus_one),
                };
                // c_s_star.ins(default, f) = c_s_star.min(*c_star_theta_i_plus_one);
                *c_star_theta_i_plus_one = c_star_theta_i_plus_one.min(*c_s);
                
            }
        }
    }
}

#[derive(Debug)]
pub enum TransitionKind {
    LastUnvisitedAction,
    LastVisitedAction,
}
