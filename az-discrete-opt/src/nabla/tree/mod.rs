use std::collections::BTreeMap;

use crate::path::{ActionPath, ActionPathFor};

use self::node::{StateNode, Transition};

use super::space::NablaStateActionSpace;

pub mod node;
pub mod update_nodes;

pub struct SearchTree<P> {
    root_node: StateNode,
    levels: Vec<BTreeMap<P, StateNode>>,
}

impl<P> SearchTree<P> {
    pub fn new<Space: NablaStateActionSpace>(
        space: &Space,
        root: &Space::State,
        cost: &Space::Cost,
        h_theta: &[f32],
        max_num_root_actions: usize,
    ) -> Self {
        Self {
            root_node: StateNode::new(space, root, cost, h_theta, max_num_root_actions).0,
            levels: Vec::new(),
        }
    }

    pub fn sizes(&self) -> impl Iterator<Item = usize> + '_ {
        self.levels.iter().map(BTreeMap::len)
    }

    pub fn roll_out_episode<Space>(
        &mut self,
        space: &Space,
        state: &mut Space::State,
        path: &mut P,
    ) -> (Vec<Transition>, NodeKind<P>)
    where
        Space: NablaStateActionSpace,
        P: Ord + ActionPath + ActionPathFor<Space>,
    {
        debug_assert_eq!(path.len(), 0);
        let Self { root_node, levels } = self;
        let transition = root_node.next_transition().unwrap();
        let a = transition.action_index();
        let action = space.action(a);
        space.act(state, &action);
        unsafe { path.push_unchecked(a) };
        let mut transitions = vec![transition];

        for level in levels.iter_mut() {
            // I hate Polonius case III
            let node = match level.contains_key(path) {
                true => level.get_mut(path).unwrap(),
                false => return (transitions, NodeKind::New(level)),
            };
            match node.next_transition() {
                Ok(trans) => {
                    let a = trans.action_index();
                    let action = space.action(a);
                    space.act(state, &action);
                    unsafe { path.push_unchecked(a) };
                    transitions.push(trans)
                },
                Err(c) => return (transitions, NodeKind::OldExhausted { c_s_t_theta_star: c })
            }
        }
        (transitions, NodeKind::NewLevel)
    }

    pub(crate) fn insert_new_node(
        &mut self,
        path: P,
        node: StateNode,
    )
    where
        P: Ord + ActionPath,
    {
        let Self {
            root_node: _,
            levels,
        } = self;
        debug_assert_eq!(path.len(), levels.len() + 1);
        let level = BTreeMap::from_iter(core::iter::once((path, node)));
        levels.push(level);
    }

    pub(crate) fn root_node(&self) -> &StateNode {
        &self.root_node
    }
}

pub enum NodeKind<'roll_out, P> {
    NewLevel,
    New(&'roll_out mut BTreeMap<P, StateNode>),
    OldExhausted { c_s_t_theta_star: f32 },
}

impl<P> NodeKind<'_, P> {
    pub fn is_new(&self) -> bool {
        !matches!(self, Self::OldExhausted { .. })
    }
}
