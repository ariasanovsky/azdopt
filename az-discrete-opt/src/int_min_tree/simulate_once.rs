use std::collections::BTreeMap;

use crate::{path::ActionPathFor, space::StateActionSpace};

use super::{
    state_data::{StateDataKind, UpperEstimateData},
    transition::INTTransition,
    INTMinTree,
};

#[derive(Debug)]
pub enum EndNodeAndLevel<'a, P> {
    NewNodeNewLevel,
    NewNodeOldLevel(&'a mut BTreeMap<P, StateDataKind>),
    OldExhaustedNode { c_t_star: f32 },
}

impl<P> INTMinTree<P> {
    pub fn simulate_once<'a, Space>(
        &'a mut self,
        space: &Space,
        s_0: &mut Space::State,
        p_0: &mut P,
        cleared_transitions: &mut Vec<INTTransition<'a>>,
        upper_estimate: impl Fn(UpperEstimateData) -> f32,
    ) -> EndNodeAndLevel<'a, P>
    where
        Space: StateActionSpace,
        P: ActionPathFor<Space> + Ord,
    {
        let Self { root_data, data } = self;
        // dbg!();
        debug_assert_eq!(
            root_data.len(),
            space.action_indices(s_0).count(),
            // "root_data.actions = {root_data.actions:?}, n_0.actions = {n_0.actions:?}",
        );
        let a_1 = root_data.best_action(&upper_estimate).unwrap();
        let action_1 = space.action(a_1.index());
        let s_i = s_0;
        let p_i = p_0;
        space.act(s_i, &action_1);
        p_i.push(space, &action_1);
        cleared_transitions.push(a_1);
        let transitions = cleared_transitions;

        for data in data.iter_mut() {
            // Polonius case III: https://github.com/rust-lang/rfcs/blob/master/text/2094-nll.md#problem-case-3-conditional-control-flow-across-functions
            /* isomorphic to
            enum PreviouslyExhaustedValue {
                NotFound,
                FoundActive,
                // the borrow checker doesn't do inference down branches of enums, it processes ownership as a stack
                FoundExhausted { c_t_star: f32 },
            }
            */
            let previously_exhausted_value: Option<Option<f32>> =
                data.get(p_i).map(|data| match data {
                    StateDataKind::Exhausted { c_t } => Some(*c_t),
                    StateDataKind::Active { data: _ } => None,
                });
            match previously_exhausted_value {
                Some(Some(c_t_star)) => return EndNodeAndLevel::OldExhaustedNode { c_t_star },
                Some(None) => {}
                None => return EndNodeAndLevel::NewNodeOldLevel(data),
            }
            let state_data = match data.get_mut(p_i) {
                Some(StateDataKind::Active { data }) => data,
                _ => unreachable!("this should be unreachable"),
            };
            // debug_assert_eq!(
            //     state_data.len(),
            //     Space::actions(s_i).count(),
            //     // "root_data.actions = {root_data.actions:?}, n_0.actions = {n_0.actions:?}",
            // );
            let a_i_plus_one = state_data.best_action(&upper_estimate).unwrap();
            let action_i_plus_1 = space.action(a_i_plus_one.index());

            // dbg!(a_i_plus_one);
            debug_assert_eq!(space.index(&action_i_plus_1), a_i_plus_one.index());
            transitions.push(a_i_plus_one);
            // debug_assert!(
            //     n_i.state().actions().any(|a| action_i_plus_1 == a),
            //     "self = {}, action_i_plus_1 = {action_i_plus_1}, actions = {:?}\ndepth = {_depth}",
            //     n_i.state(),
            //     n_i.state().actions().map(|a| a.index()).collect::<Vec<_>>(),
            // );

            // debug_assert!(
            //     s_i.actions().any(|a| action_i_plus_1 == a),
            //     "self = {s_i}, action_i_plus_1 = {action_i_plus_1}, actions = {:?}\np_i = {p_i:?}\ndepth = {_depth}",
            //     s_i.actions().map(|a| a.index()).collect::<Vec<_>>(),
            // );
            space.act(s_i, &action_i_plus_1);
            p_i.push(space, &action_i_plus_1);
        }
        EndNodeAndLevel::NewNodeNewLevel
    }
}
