use std::collections::BTreeSet;

use petgraph::visit::EdgeRef;

use crate::path::ActionPath;

use super::{NodeIndex, SearchTree};

impl SearchTree {
    pub(crate) fn find_node<P: ActionPath>(&self, target: &P) -> Option<NodeIndex> {
        let mut source = P::new();
        let root_id = NodeIndex::default();
        let mut nodes_seen = BTreeSet::from_iter(core::iter::once(root_id));
        let mut actions_taken = vec![];
        let mut neighborhoods = vec![self.tree.edges_directed(root_id, petgraph::Direction::Outgoing)];
        while let Some(last_neighborhood) = neighborhoods.last_mut() {
            let next_action = last_neighborhood.find(|e| {
                let action_pos = e.weight().prediction_pos;
                let action = self.predictions[action_pos].a_id;
                if source.extends_towards(action, target) {
                    let inserted = nodes_seen.insert(e.target());
                    inserted
                } else {
                    false
                }
            });
            match next_action {
                Some(e) => {
                    let next_node = e.target();
                    let action_pos = e.weight().prediction_pos;
                    let action = self.predictions[action_pos].a_id;
                    unsafe { source.push_unchecked(action) };
                    if source.len() == target.len() {
                        return Some(next_node);
                    }
                    actions_taken.push(action);
                    let next_neighborhood = self.tree.edges_directed(next_node, petgraph::Direction::Outgoing);
                    neighborhoods.push(next_neighborhood);
                },
                None => match actions_taken.pop() {
                    Some(last_action) => {
                        unsafe { source.undo_unchecked(last_action) };
                        let _last = neighborhoods.pop();
                        debug_assert!(_last.is_some());
                    },
                    None => {
                        debug_assert!(source.is_empty());
                        debug_assert!(neighborhoods.len() == 1);
                        return None;
                    },
                },
            }
        }
        None
    }

    pub(crate) fn find_path<P: ActionPath>(&self, target: NodeIndex) -> Option<P> {
        let mut source = P::new();
        let root_id = NodeIndex::default();
        if target == root_id {
            return Some(source);
        }
        let mut nodes_seen = BTreeSet::from_iter(core::iter::once(root_id));
        let mut actions_taken = vec![];
        let mut neighborhoods = vec![self.tree.edges_directed(root_id, petgraph::Direction::Outgoing)];
        while let Some(last_neighborhood) = neighborhoods.last_mut() {
            let next_action = last_neighborhood.find(|e| {
                let t = e.target();
                if t <= target {
                    let inserted = nodes_seen.insert(e.target());
                    inserted
                } else {
                    false
                }
            });
            match next_action {
                Some(e) => {
                    let next_node = e.target();
                    let action_pos = e.weight().prediction_pos;
                    let action = self.predictions[action_pos].a_id;
                    unsafe { source.push_unchecked(action) };
                    if next_node == target {
                        return Some(source);
                    }
                    actions_taken.push(action);
                    let next_neighborhood = self.tree.edges_directed(next_node, petgraph::Direction::Outgoing);
                    neighborhoods.push(next_neighborhood);
                },
                None => match actions_taken.pop() {
                    Some(last_action) => {
                        unsafe { source.undo_unchecked(last_action) };
                        let _last = neighborhoods.pop();
                        debug_assert!(_last.is_some());
                    },
                    None => {
                        debug_assert!(source.is_empty());
                        debug_assert!(neighborhoods.len() == 1);
                        dbg!();
                        return None;
                    }
                },
            }
        }
        dbg!();
        None
    }
}