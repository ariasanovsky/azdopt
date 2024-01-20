use std::collections::BTreeMap;

use crate::{
    nabla::{space::NablaStateActionSpace, tree::NodeIndex},
    path::{ActionPath, ActionPathFor},
};

use super::{EdgeIndex, SearchTree};

pub(crate) struct CascadeTracker {
    current_nodes: BTreeMap<NodeIndex, bool>,
    next_nodes: BTreeMap<NodeIndex, bool>,
}

impl CascadeTracker {
    fn new(state_pos: NodeIndex, child_active: bool) -> Self {
        Self {
            current_nodes: core::iter::once((state_pos, child_active)).collect(),
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

impl<P> SearchTree<P> {
    pub(crate) fn cascade_updates(&mut self, edge_id: EdgeIndex) {
        let edge = &self.tree.raw_edges()[edge_id.index()];
        let child_pos = edge.target();
        let parent_pos = edge.source();
        let child_weight = &mut self.tree[child_pos];
        let s_t_active = match &mut child_weight.n_t {
            Some(n_t) => {
                *n_t = n_t.checked_add(1).unwrap();
                true
            },
            None => false,
        };
        let child_c_star = child_weight.c_t_star;
        let aparent_c_star = &mut self.tree[parent_pos].c_t_star;
        *aparent_c_star = aparent_c_star.min(child_c_star);
        let mut cascade_tracker = CascadeTracker::new(parent_pos, s_t_active);
        while let Some((child_pos, seen_active_child)) = cascade_tracker.pop_front() {
            let n_t = match seen_active_child {
                true => Some(self.tree[child_pos].n_t.as_mut().expect("cannot visit an exhausted node")),
                false => {self.update_exhaustion(child_pos)},
            };
            let child_is_active = match n_t {
                Some(n_t) => {
                    *n_t = n_t.checked_add(1).unwrap();
                    true
                },
                None => false,
            };
            let mut neigh = self.tree.neighbors_directed(child_pos, petgraph::Direction::Incoming).detach();
            while let Some(parent_pos) = neigh.next_node(&self.tree) {
                let parent_weight = &mut self.tree[parent_pos];
                let parent_c_star = &mut parent_weight.c_t_star;
                *parent_c_star = parent_c_star.min(child_c_star);
                cascade_tracker.upsert(parent_pos, child_is_active);
            }
            // todo!();
        }
    }

    pub(crate) fn _clear_path<Space>(
        &mut self,
        space: &Space,
        root: &Space::State,
        state: &mut Space::State,
        path: &mut P,
        state_pos: &mut NodeIndex,
    ) where
        Space: NablaStateActionSpace,
        Space::State: Clone,
        P: Ord + ActionPath + ActionPathFor<Space> + Clone,
    {
        // debug_assert!(!transitions.is_empty());
        debug_assert!(self.positions.contains_key(path));

        let mut reached_root = false;
        todo!();
        let mut decay_tracker: CascadeTracker = todo!(); //DecayTracker::new(state_pos.unwrap());
        while let Some(child_pos) = decay_tracker.pop_front() {
            // dbg!(child_pos);
            todo!();
            // let c_as = self.nodes[child_pos.get()].c_star;
            // let c_as_star = self.nodes[child_pos.get()].c_star;
            // let child_exhausted = self.nodes[child_pos.get()].is_exhausted();
            // let parents = self.in_neighborhoods.get(child_pos.get()).unwrap();
            // match child_exhausted {
            //     true => for Transition {
            //         state_position: parent_pos,
            //         action_position,
            //     } in parents {
            //         // dbg!(parent_pos);
            //         let parent_node = &mut self.nodes[*parent_pos];
            //         parent_node.exhaust(*action_position, c_as_star);
            //         if let Some(parent_pos) = NonZeroUsize::new(*parent_pos) {
            //             decay_tracker.insert(parent_pos);
            //         } else {
            //             reached_root = true;
            //         }
            //     },
            //     false => for Transition {
            //         state_position: parent_pos,
            //         action_position,
            //     } in parents {
            //         // dbg!(parent_pos);
            //         let parent_node = &mut self.nodes[*parent_pos];
            //         let c_s = parent_node.c;
            //         let h_t_sa = space.h_sa(c_s, c_as, c_as_star);
            //         parent_node.update_h_t_sa(*action_position, c_as_star, h_t_sa);
            //         if let Some(parent_pos) = NonZeroUsize::new(*parent_pos) {
            //             decay_tracker.insert(parent_pos);
            //         } else {
            //             reached_root = true;
            //         }
            //     },
            // }
            // let children = self.in_neighborhoods.get_mut(parent_pos.get()).unwrap();
            // children.retain(
            //     |Transition {
            //          state_position,
            //          action_position,
            //      }| {
            //         let child_node = &self.nodes[*state_position];
            //         let action_data = &child_node.actions[*action_position];
            //         action_data.g_sa().is_some()
            //     },
            // );
            // let num_children = children.len();
            // let num_children = NonZeroUsize::new(num_children).expect("No children");
            // for Transition {
            //     state_position: child_pos,
            //     action_position,
            // } in children
            // {
            //     let child_node = &mut self.nodes[*child_pos];
            //     child_node.update_with_parent_c_star(*action_position, parent_c_star, decay);
            //     match parent_exhausted {
            //         true => {
            //             child_node.exhaust_action(*action_position);
            //         }
            //         false => {
            //             child_node.update_c_star(parent_c_star, *action_position, parent_decay.0);
            //         }
            //     }
            //     if let Some(child_pos) = NonZeroUsize::new(*child_pos) {
            //         decay_tracker.insert_or_update(
            //             child_pos,
            //             Decay(parent_decay.0 / num_children.get() as f32),
            //         );
            //     } else {
            //         reached_root = true;
            //     }
            //     todo!();
            // }
            // todo!();
        }

        debug_assert!(reached_root);
        // transitions.clear();
        // debug_assert_ne!(transitions.capacity(), 0);
        todo!();
        // path.clear();
        // state.clone_from(root);
        // *state_pos = None;
    }
}
