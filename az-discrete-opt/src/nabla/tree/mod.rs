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
        s_0: Space::State,
        c_0: &Space::Cost,
        h_theta: &[f32],
    ) -> Self {
        Self {
            root_node: match StateNodeKind::new(space, s_0, c_0, h_theta) {
                _ => todo!(),
            },
            nodes: Vec::new(),
        }
    }
}