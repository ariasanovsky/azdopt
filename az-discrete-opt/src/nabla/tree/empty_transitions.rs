use std::collections::{BTreeMap, BTreeSet};

use crate::nabla::tree::NodeIndex;

use super::{EdgeIndex, SearchTree, state_weight::NumLeafDescendants};

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
        let foo = self.next_nodes.entry(state_pos);
        let foo = foo.and_modify(|active_child_seen| {
            *active_child_seen = *active_child_seen || child_active;
        }).or_insert(child_active);
        *foo
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

    fn upsert(&mut self, state_pos: NodeIndex, num_child_leaf_descendants: NumLeafDescendants) -> NumLeafDescendants {
        let foo = self.next_nodes.entry(state_pos);
        let foo = foo.and_modify(|active_child_seen| {
            *active_child_seen = active_child_seen.join(&num_child_leaf_descendants);
        }).or_insert(num_child_leaf_descendants);
        *foo
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
                },
            };
            child_node_weight.n_t.increment();
            let child_c_t_star = child_node_weight.c_t_star;
            let mut neigh = self.tree.neighbors_directed(child_id, petgraph::Direction::Incoming).detach();
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
            let mut neigh = self.tree.neighbors_directed(child_index, petgraph::Direction::Incoming).detach();
            while let Some(parent_index) = neigh.next_node(&self.tree) {
                let parent_weight = &mut self.tree[parent_index];
                parent_weight.c_t_star = parent_weight.c_t_star.min(child_c_t_star);
                cascade_tracker.upsert(parent_index, child_n_t);
            }
            // todo!()
        }
    }

    pub(crate) fn _cascade_updates(&mut self, edge_id: EdgeIndex) {
        let edge = &self.tree.raw_edges()[edge_id.index()];
        let child_pos = edge.target();
        let parent_pos = edge.source();
        let child_weight = &mut self.tree[child_pos];
        let s_t_num_leaf_descendants = child_weight.n_t;
        let new_leaf = s_t_num_leaf_descendants.value() == 0;
        let child_c_star = child_weight.c_t_star;
        let aparent_c_star = &mut self.tree[parent_pos].c_t_star;
        *aparent_c_star = aparent_c_star.min(child_c_star);
        let mut cascade_tracker = CascadeTracker::new(parent_pos, s_t_num_leaf_descendants);
        while let Some((child_pos, join_num_leaf_descendants)) = cascade_tracker.pop_front() {
            todo!();
            let foo = match join_num_leaf_descendants.try_active() {
                Some(num) => {
                    // &mut self.tree[child_pos].n_t
                    todo!()
                },
                None => {
                    self.update_exhaustion(child_pos)
                },
            };
            todo!();
            // let n_t = match seen_active_child {
            //     true => Some(self.tree[child_pos].n_t.as_mut().expect("cannot visit an exhausted node")),
            //     false => {self.update_exhaustion(child_pos)},
            // };
            // let child_is_active = match n_t {
            //     Some(n_t) => {
            //         *n_t = n_t.checked_add(1).unwrap();
            //         true
            //     },
            //     None => false,
            // };
            let child_c_star = self.tree[child_pos].c_t_star;
            let mut neigh = self.tree.neighbors_directed(child_pos, petgraph::Direction::Incoming).detach();
            while let Some(parent_pos) = neigh.next_node(&self.tree) {
                let parent_weight = &mut self.tree[parent_pos];
                let parent_c_star = &mut parent_weight.c_t_star;
                *parent_c_star = parent_c_star.min(child_c_star);
                cascade_tracker.upsert(parent_pos, todo!());
            }
            todo!();
        }
    }

    // pub(crate) fn _clear_path<Space>(
    //     &mut self,
    //     space: &Space,
    //     root: &Space::State,
    //     state: &mut Space::State,
    //     path: &mut P,
    //     state_pos: &mut NodeIndex,
    // ) where
    //     Space: NablaStateActionSpace,
    //     Space::State: Clone,
    //     P: Ord + ActionPath + ActionPathFor<Space> + Clone,
    // {
    //     // debug_assert!(!transitions.is_empty());
    //     debug_assert!(self.positions.contains_key(path));

    //     let mut reached_root = false;
    //     todo!();
    //     let mut decay_tracker: CascadeTracker = todo!(); //DecayTracker::new(state_pos.unwrap());
    //     while let Some(child_pos) = decay_tracker.pop_front() {
    //         // dbg!(child_pos);
    //         todo!();
    //         // let c_as = self.nodes[child_pos.get()].c_star;
    //         // let c_as_star = self.nodes[child_pos.get()].c_star;
    //         // let child_exhausted = self.nodes[child_pos.get()].is_exhausted();
    //         // let parents = self.in_neighborhoods.get(child_pos.get()).unwrap();
    //         // match child_exhausted {
    //         //     true => for Transition {
    //         //         state_position: parent_pos,
    //         //         action_position,
    //         //     } in parents {
    //         //         // dbg!(parent_pos);
    //         //         let parent_node = &mut self.nodes[*parent_pos];
    //         //         parent_node.exhaust(*action_position, c_as_star);
    //         //         if let Some(parent_pos) = NonZeroUsize::new(*parent_pos) {
    //         //             decay_tracker.insert(parent_pos);
    //         //         } else {
    //         //             reached_root = true;
    //         //         }
    //         //     },
    //         //     false => for Transition {
    //         //         state_position: parent_pos,
    //         //         action_position,
    //         //     } in parents {
    //         //         // dbg!(parent_pos);
    //         //         let parent_node = &mut self.nodes[*parent_pos];
    //         //         let c_s = parent_node.c;
    //         //         let h_t_sa = space.h_sa(c_s, c_as, c_as_star);
    //         //         parent_node.update_h_t_sa(*action_position, c_as_star, h_t_sa);
    //         //         if let Some(parent_pos) = NonZeroUsize::new(*parent_pos) {
    //         //             decay_tracker.insert(parent_pos);
    //         //         } else {
    //         //             reached_root = true;
    //         //         }
    //         //     },
    //         // }
    //         // let children = self.in_neighborhoods.get_mut(parent_pos.get()).unwrap();
    //         // children.retain(
    //         //     |Transition {
    //         //          state_position,
    //         //          action_position,
    //         //      }| {
    //         //         let child_node = &self.nodes[*state_position];
    //         //         let action_data = &child_node.actions[*action_position];
    //         //         action_data.g_sa().is_some()
    //         //     },
    //         // );
    //         // let num_children = children.len();
    //         // let num_children = NonZeroUsize::new(num_children).expect("No children");
    //         // for Transition {
    //         //     state_position: child_pos,
    //         //     action_position,
    //         // } in children
    //         // {
    //         //     let child_node = &mut self.nodes[*child_pos];
    //         //     child_node.update_with_parent_c_star(*action_position, parent_c_star, decay);
    //         //     match parent_exhausted {
    //         //         true => {
    //         //             child_node.exhaust_action(*action_position);
    //         //         }
    //         //         false => {
    //         //             child_node.update_c_star(parent_c_star, *action_position, parent_decay.0);
    //         //         }
    //         //     }
    //         //     if let Some(child_pos) = NonZeroUsize::new(*child_pos) {
    //         //         decay_tracker.insert_or_update(
    //         //             child_pos,
    //         //             Decay(parent_decay.0 / num_children.get() as f32),
    //         //         );
    //         //     } else {
    //         //         reached_root = true;
    //         //     }
    //         //     todo!();
    //         // }
    //         // todo!();
    //     }

    //     debug_assert!(reached_root);
    //     // transitions.clear();
    //     // debug_assert_ne!(transitions.capacity(), 0);
    //     todo!();
    //     // path.clear();
    //     // state.clone_from(root);
    //     // *state_pos = None;
    // }
}
