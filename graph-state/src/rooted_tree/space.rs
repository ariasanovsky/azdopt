use crate::simple_graph::edge::Edge;

use super::RootedOrderedTree;

pub struct OrderedEdge {
    edge: Edge,
}

impl<const N: usize> RootedOrderedTree<N> {
    pub fn set_root(&mut self, ordered_edge: OrderedEdge) {
        todo!()
    }
}