use std::collections::BTreeMap;

use crate::{nabla::tree::node::ActionData, path::{ActionPath, ActionPathFor}};

use self::node::{StateNode};

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
    ) -> () //(NewNodeLevel<P>, Transitions)
    where
        Space: NablaStateActionSpace,
        P: ActionPath + ActionPathFor<Space>,
    {
        let Self { root_node, nodes } = self;
        let (a, g_sa, kind_sa) = root_node.next_action_data().unwrap();
        let action = space.action(a);
        space.act(state, &action);
        todo!("path.push(*a)");
        todo!("*n_sa += 1");
        todo!()
    }
}

pub struct Transitions;

// pub enum NewNodeLevel<'roll_out, P> {
//     New,
//     Old(&'roll_out mut BTreeMap<P, StateNodeKind>),
// }