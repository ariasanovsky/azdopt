use std::collections::{BTreeMap, VecDeque};

use crate::path::{ActionPath, ActionPathFor};

use self::node::{StateNode, ActionData, Transition};

use super::space::NablaStateActionSpace;

pub mod node;
pub mod update_nodes;

pub struct SearchTree<P> {
    root_node: StateNode,
    nodes: Vec<BTreeMap<P, StateNode>>,
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
            root_node: StateNode::new(space, root, cost, h_theta, max_num_root_actions),
            nodes: Vec::new(),
        }
    }

    pub fn roll_out_episode<Space>(
        &mut self,
        space: &Space,
        state: &mut Space::State,
        path: &mut P,
    ) -> (Vec<Transition>, NewNodeKind<P>) //(NewNodeLevel<P>, Transitions)
    where
        Space: NablaStateActionSpace,
        P: ActionPath + ActionPathFor<Space>,
    {
        let Self { root_node, nodes } = self;
        let transition = root_node.next_transition().unwrap();
        let a = transition.action_index();
        let action = space.action(a);
        space.act(state, &action);
        unsafe { path.push_unchecked(a) };
        let mut transitions = vec![transition];
        for (i, level) in nodes.iter_mut().enumerate() {
            todo!()
        }
        (
            transitions,
            NewNodeKind::NewLevel,
        )
    }
}

pub enum NewNodeKind<'roll_out, P> {
    NewLevel,
    OldLevelNewNode(&'roll_out mut BTreeMap<P, StateNode>),
    OldExhaustedNode { c_s_t_theta_star: f32 },
}

impl<P> NewNodeKind<'_, P> {
    pub fn is_new(&self) -> bool {
        !matches!(self, Self::OldExhaustedNode { .. })
    }
}
