use std::collections::BTreeMap;

use crate::path::{ActionPath, ActionPathFor};

use self::node::{StateNode, ActionDataKind};

use super::space::NablaStateActionSpace;

mod node;

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
        let (c_s, a, g_sa, kind_sa) = root_node.next_action_data().unwrap();
        let action = space.action(a);
        space.act(state, &action);
        unsafe { path.push_unchecked(a) };
        let mut transitions = vec![Transition {
            c_s,
            g_theta_star_sa: g_sa,
            kind_sa,
        }];
        for (i, level) in nodes.iter_mut().enumerate() {
            todo!()
        }
        (
            transitions,
            NewNodeKind::NewLevel,
        )
    }
}

pub struct Transition<'roll_out> {
    c_s: f32,
    g_theta_star_sa: &'roll_out mut f32,
    kind_sa: &'roll_out mut ActionDataKind,
}

pub enum NewNodeKind<'roll_out, P> {
    NewLevel,
    OldLevelNewNode(&'roll_out mut BTreeMap<P, StateNode>),
    OldExhaustedNode { c_s_star: f32 },
}