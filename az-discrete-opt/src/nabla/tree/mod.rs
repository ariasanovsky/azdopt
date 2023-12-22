use std::collections::BTreeMap;

use self::node::{StateNode, StateNodeKind};

use super::space::NablaStateActionSpace;

mod node;

pub struct SearchTree<P> {
    root_node: StateNode,
    nodes: Vec<BTreeMap<P, StateNodeKind>>,
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
            root_node: match StateNodeKind::new(space, root, cost, h_theta, max_num_root_actions) {
                _ => todo!(),
            },
            nodes: Vec::new(),
        }
    }
}