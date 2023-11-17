use std::collections::BTreeMap;

use crate::{tree_node::{TreeNode, TreeNodeFor}, space::StateActionSpace, path::ActionPathFor};

use super::{transition::INTTransition, INTMinTree, state_data::StateDataKind};

pub(crate) enum EndNodeAndLevel<'a, P> {
    NewNodeNewLevel,
    NewNodeOldLevel(&'a mut BTreeMap<P, StateDataKind>),
    OldExhaustedNode { c_t_star: f32 },
}

pub struct INTTransitions<'a, P> {
    pub(crate) a_1: INTTransition<'a>,
    pub(crate) transitions: Vec<INTTransition<'a>>,
    pub(crate) end: EndNodeAndLevel<'a, P>,
}

impl<P> INTMinTree<P> {
    pub fn simulate_once<'a, Space>(
        &'a mut self,
        n_0: &mut (impl TreeNode<Path = P, State = Space::State> + TreeNodeFor<Space>),
    ) -> INTTransitions<'a, P>
    where
        // Space::Action: Action<Space>,
        Space: StateActionSpace,
        P: ActionPathFor<Space> + Ord,
        // N: TreeNode<Path = P> + TreeNodeFor<Space>,
        // N: TreeNode<Path = P>,
        // N::Action: Action<N>,
        // N::Path: Ord,
    {
        let Self { root_data, data } = self;
        debug_assert_eq!(
            root_data.visited_actions.len(),
            Space::actions(n_0.state()).count(),
            // "root_data.actions = {root_data.actions:?}, n_0.actions = {n_0.actions:?}",
        );
        let a_1 = root_data.best_action().unwrap();
        let action_1 = Space::from_index(a_1.index());
        let n_i = n_0;
        n_i.apply_action(&action_1);
        // n_i.apply_action(&action_1);
        // unsafe { n_i.state().act_unchecked(&action_1) };
        // let p_i = p_0;
        // p_i.push(&action_1);
        let mut transitions: Vec<_> = vec![];

        for (_depth, data) in data.iter_mut().enumerate() {
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
                data.get(n_i.path()).map(|data| match data {
                    StateDataKind::Exhausted { c_t_star } => Some(*c_t_star),
                    StateDataKind::Active { data: _ } => None,
                });
            match previously_exhausted_value {
                Some(Some(c_t_star)) => {
                    let end = EndNodeAndLevel::OldExhaustedNode { c_t_star };
                    return INTTransitions {
                        a_1,
                        transitions,
                        end,
                    };
                }
                Some(None) => {}
                None => {
                    let end = EndNodeAndLevel::NewNodeOldLevel(data);
                    return INTTransitions {
                        a_1,
                        transitions,
                        end,
                    };
                }
            }
            let state_data = match data.get_mut(n_i.path()) {
                Some(StateDataKind::Active { data }) => data,
                _ => unreachable!("this should be unreachable"),
            };
            debug_assert_eq!(
                state_data.visited_actions.len(),
                Space::actions(n_i.state()).count(),
                // "root_data.actions = {root_data.actions:?}, n_0.actions = {n_0.actions:?}",
            );
            let a_i_plus_one = state_data.best_action().unwrap();
            let action_i_plus_1 = Space::from_index(a_i_plus_one.index());

            // dbg!(a_i_plus_one);
            debug_assert_eq!(Space::index(&action_i_plus_1), a_i_plus_one.index());
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

            transitions.push(a_i_plus_one);
            n_i.apply_action(&action_i_plus_1);
        }
        INTTransitions {
            a_1,
            transitions,
            end: EndNodeAndLevel::NewNodeNewLevel,
        }
    }
}