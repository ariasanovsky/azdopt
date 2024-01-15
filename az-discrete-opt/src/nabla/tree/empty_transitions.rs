use core::num::NonZeroUsize;
use std::collections::BTreeSet;

use crate::{
    nabla::space::NablaStateActionSpace,
    path::{ActionPath, ActionPathFor},
};

use super::{SearchTree, Transition};

// struct Decay(f32);

pub(crate) struct DecayTracker {
    current_nodes: BTreeSet<NonZeroUsize>,
    next_nodes: BTreeSet<NonZeroUsize>,
}

impl DecayTracker {
    fn new(state_pos: NonZeroUsize) -> Self {
        Self {
            current_nodes: core::iter::once(state_pos).collect(),
            next_nodes: BTreeSet::new(),
        }
    }

    fn pop_front(&mut self) -> Option<NonZeroUsize> {
        self.current_nodes.pop_first().or_else(|| {
            core::mem::swap(&mut self.current_nodes, &mut self.next_nodes);
            self.current_nodes.pop_first()
        })
    }

    fn insert(&mut self, state_pos: NonZeroUsize) -> bool {
        self.next_nodes
            .insert(state_pos)
    }
}

impl<P> SearchTree<P> {
    pub(crate) fn clear_path<Space>(
        &mut self,
        root: &Space::State,
        state: &mut Space::State,
        path: &mut P,
        // transitions: &mut Vec<Transition>,
        state_pos: &mut Option<NonZeroUsize>,
        decay: f32,
    ) where
        Space: NablaStateActionSpace,
        Space::State: Clone,
        P: Ord + ActionPath + ActionPathFor<Space> + Clone,
    {
        // debug_assert!(!transitions.is_empty());
        debug_assert!(self.positions.contains_key(path));

        let mut reached_root = false;

        let mut decay_tracker = DecayTracker::new(state_pos.unwrap());
        while let Some(child_pos) = decay_tracker.pop_front() {
            // dbg!(child_pos);
            let child_c_star = self.nodes[child_pos.get()].c_star;
            let child_exhausted = self.nodes[child_pos.get()].is_exhausted();
            let parents = self.in_neighborhoods.get(child_pos.get()).unwrap();
            match child_exhausted {
                true => for Transition {
                    state_position: parent_pos,
                    action_position,
                } in parents {
                    // dbg!(parent_pos);
                    let parent_node = &mut self.nodes[*parent_pos];
                    parent_node.update_c_star_and_exhaust(*action_position, child_c_star);
                    if let Some(parent_pos) = NonZeroUsize::new(*parent_pos) {
                        decay_tracker.insert(parent_pos);
                    } else {
                        reached_root = true;
                    }
                },
                false => for Transition {
                    state_position: parent_pos,
                    action_position,
                } in parents {
                    // dbg!(parent_pos);
                    let parent_node = &mut self.nodes[*parent_pos];
                    parent_node.update_c_star_and_decay(*action_position, child_c_star, decay);
                    if let Some(parent_pos) = NonZeroUsize::new(*parent_pos) {
                        decay_tracker.insert(parent_pos);
                    } else {
                        reached_root = true;
                    }
                },
            }
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
        path.clear();
        state.clone_from(root);
        *state_pos = None;
    }
}
