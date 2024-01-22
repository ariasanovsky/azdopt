use std::collections::BTreeMap;

use crate::nabla::tree::NodeIndex;

use super::{state_weight::NumLeafDescendants, EdgeIndex, SearchTree};

pub(crate) struct CascadeTracker {
    current_nodes: BTreeMap<NodeIndex, NumLeafDescendants>,
    next_nodes: BTreeMap<NodeIndex, NumLeafDescendants>,
}

struct NodeTracker {
    current_nodes: BTreeMap<NodeIndex, bool>,
    next_nodes: BTreeMap<NodeIndex, bool>,
}

impl NodeTracker {
    fn new(state_pos: NodeIndex) -> Self {
        Self {
            current_nodes: core::iter::once((state_pos, false)).collect(),
            next_nodes: BTreeMap::new(),
        }
    }

    fn pop_front(&mut self) -> Option<(NodeIndex, bool)> {
        self.current_nodes.pop_first().or_else(|| {
            core::mem::swap(&mut self.current_nodes, &mut self.next_nodes);
            self.current_nodes.pop_first()
        })
    }

    fn upsert(&mut self, state_pos: NodeIndex, child_active: bool) -> bool {
        let entry = self.next_nodes.entry(state_pos);
        let value = entry
            .and_modify(|active_child_seen| {
                *active_child_seen = *active_child_seen || child_active;
            })
            .or_insert(child_active);
        *value
    }
}

impl CascadeTracker {
    fn new(state_pos: NodeIndex, num_child_leaf_descendants: NumLeafDescendants) -> Self {
        Self {
            current_nodes: core::iter::once((state_pos, num_child_leaf_descendants)).collect(),
            next_nodes: BTreeMap::new(),
        }
    }

    fn pop_front(&mut self) -> Option<(NodeIndex, NumLeafDescendants)> {
        self.current_nodes.pop_first().or_else(|| {
            core::mem::swap(&mut self.current_nodes, &mut self.next_nodes);
            self.current_nodes.pop_first()
        })
    }

    fn upsert(
        &mut self,
        state_pos: NodeIndex,
        num_child_leaf_descendants: NumLeafDescendants,
    ) -> NumLeafDescendants {
        let entry = self.next_nodes.entry(state_pos);
        let value = entry
            .and_modify(|active_child_seen| {
                *active_child_seen = active_child_seen.join(&num_child_leaf_descendants);
            })
            .or_insert(num_child_leaf_descendants);
        *value
    }
}

impl<P> SearchTree<P> {
    pub(crate) fn cascade_new_terminal(&mut self, edge_id: EdgeIndex) {
        let a_t = &self.tree.raw_edges()[edge_id.index()];
        let s_t = &self.tree[a_t.target()];
        debug_assert_eq!(s_t.n_t.value(), 0);
        debug_assert!(!s_t.n_t.is_active());
        let c_t = s_t.c_t_star;
        let parent_index = a_t.source();
        let parent_node = &mut self.tree[parent_index];
        parent_node.c_t_star = parent_node.c_t_star.min(c_t);
        let mut node_tracker = NodeTracker::new(parent_index);
        while let Some((child_id, active_ancestor_seen)) = node_tracker.pop_front() {
            let child_node_weight = match active_ancestor_seen {
                true => &mut self.tree[child_id],
                false => {
                    let weight = self.update_exhaustion(child_id);
                    // todo!();
                    weight
                }
            };
            child_node_weight.n_t.increment();
            let child_c_t_star = child_node_weight.c_t_star;
            let mut neigh = self
                .tree
                .neighbors_directed(child_id, petgraph::Direction::Incoming)
                .detach();
            while let Some(parent_id) = neigh.next_node(&self.tree) {
                let parent_node = &mut self.tree[parent_id];
                parent_node.c_t_star = parent_node.c_t_star.min(child_c_t_star);
                node_tracker.upsert(parent_id, parent_node.n_t.is_active());
            }
        }
    }

    pub(crate) fn cascade_old_node(&mut self, edge_id: EdgeIndex) {
        let a_t = &self.tree.raw_edges()[edge_id.index()];
        let s_t = &self.tree[a_t.target()];
        let c_t_star = s_t.c_t_star;
        let parent_index = a_t.source();
        let mut cascade_tracker = CascadeTracker::new(parent_index, s_t.n_t);
        let parent_weight = &mut self.tree[parent_index];
        parent_weight.c_t_star = parent_weight.c_t_star.min(c_t_star);
        while let Some((child_index, num_child_leaf_descendants)) = cascade_tracker.pop_front() {
            let child_weight = match num_child_leaf_descendants.is_active() {
                true => &mut self.tree[child_index],
                false => self.update_exhaustion(child_index),
            };
            child_weight.n_t = child_weight.n_t.join(&num_child_leaf_descendants);
            let child_c_t_star = child_weight.c_t_star;
            let child_n_t = child_weight.n_t;
            let mut neigh = self
                .tree
                .neighbors_directed(child_index, petgraph::Direction::Incoming)
                .detach();
            while let Some(parent_index) = neigh.next_node(&self.tree) {
                let parent_weight = &mut self.tree[parent_index];
                parent_weight.c_t_star = parent_weight.c_t_star.min(child_c_t_star);
                cascade_tracker.upsert(parent_index, child_n_t);
            }
            // todo!()
        }
    }
}
