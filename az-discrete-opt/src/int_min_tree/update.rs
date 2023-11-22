use crate::{
    int_min_tree::{
        simulate_once::{EndNodeAndLevel, INTTransitions},
        state_data::StateDataKind,
        NewTreeLevel,
    },
    path::ActionPathFor,
    space::StateActionSpace,
    state::cost::Cost,
};

use super::transition::INTTransition;

impl<'a, P> INTTransitions<'a, P> {
    pub fn update_existing_nodes<Space>(
        self,
        c_t: &impl Cost<f32>,
        s_t: &Space::State,
        probs_t: &[f32],
        g_star_theta_s_t: &[f32],
        transitions: &mut [INTTransition<'a>],
    ) -> Option<NewTreeLevel<P>>
    where
        Space: StateActionSpace,
        P: ActionPathFor<Space> + Ord,
        P: Ord + Clone,
    {
        debug_assert_eq!(g_star_theta_s_t.len(), 1);
        let INTTransitions {
            end,
            p_t,
        } = self;
        let c_t = c_t.evaluate();
        enum CStarEndState {
            Terminal(f32),
            Active(f32),
        }
        let (c_t_star, new_level): (CStarEndState, Option<NewTreeLevel<P>>) = match end {
            EndNodeAndLevel::NewNodeNewLevel => {
                // create data to insert at the new level
                let new_data: StateDataKind = StateDataKind::new::<Space>(probs_t, c_t, s_t);
                // if not terminal, use the model to predict the min cost, should the path continue
                let c_star = match &new_data {
                    StateDataKind::Exhausted { c_t } => CStarEndState::Terminal(*c_t),
                    StateDataKind::Active { data: _ } => {
                        CStarEndState::Active(c_t - g_star_theta_s_t[0].max(0.0))
                    }
                };
                // create a new level
                let new_level = NewTreeLevel {
                    p_t: p_t.clone(),
                    data_t: new_data,
                };
                (c_star, Some(new_level))
            }
            EndNodeAndLevel::NewNodeOldLevel(old_level) => {
                // create data to insert at the new level
                let new_data: StateDataKind = StateDataKind::new::<Space>(probs_t, c_t, s_t);
                // if not terminal, use the model to predict the min cost, should the path continue
                let c_star = match &new_data {
                    StateDataKind::Exhausted { c_t } => CStarEndState::Terminal(*c_t),
                    StateDataKind::Active { data: _ } => {
                        CStarEndState::Active(c_t - g_star_theta_s_t[0].max(0.0))
                    }
                };
                // insert the new data at the old level
                let _i = old_level.insert(p_t.clone(), new_data);
                debug_assert!(_i.is_none(), "this node should not already exist");
                (c_star, None)
            }
            EndNodeAndLevel::OldExhaustedNode { c_t_star } => {
                // we make no updates to `s_t`'s node and do not use the model to adjust c_t
                (CStarEndState::Terminal(c_t_star), None)
            }
        };
        let (a_1, transitions) = transitions
            .split_first_mut()
            .expect("there should be at least one transition");
        let transitions = transitions.iter_mut().rev();
        // a cascade begins if the transition s_{t-1} -[a_t]-> s_t has s_t exhausted (necessarily, s_{t-1} is active)
        // since a_t leads to an exhausted node, we update s_{t-1}'s action data by removing a_t
        // if this exhausts s_{t-1}, s_{t-1} is marked as exhausted and the cascade continues
        // during the cascade, we
        // todo! debug the cascade
        let mut c_i_star = match c_t_star {
            CStarEndState::Terminal(c) => c,
            CStarEndState::Active(c) => c,
        };
        // let mut c_i_star = match c_t_star {
        //     CStarEndState::Terminal(c_t_star) => {
        //         let mut c_i_star = c_t_star;
        //         for a_i_plus_one in transitions.by_ref() {
        //             if !a_i_plus_one.cascade_update(&mut c_i_star) {
        //                 break;
        //             }
        //         }
        //         c_i_star
        //     }
        //     CStarEndState::Active(c_t_star) => c_t_star,
        // };
        transitions.for_each(|a_i_plus_one| {
            a_i_plus_one.update(&mut c_i_star);
        });
        // // todo! also backpropagate exhaustion
        // let mut c_star_theta_i = c_t.cost() - h_star_theta_s_t.max(0.0);
        // transitions.into_iter().rev().for_each(|a_i_plus_one| {
        //     // let g_star_theta_i = c_i - c_star_theta_i;
        //     let c_i_star = a_i_plus_one.c_i_star();
        //     a_i_plus_one.update(c_star_theta_i);
        //     c_star_theta_i = c_star_theta_i.min(c_i_star);
        //     // ");
        // });
        a_1.update(&mut c_i_star);
        // node
        new_level
    }
}
