use std::collections::{BTreeMap, BTreeSet};

use core::num::NonZeroUsize;

use crate::path::{ActionPath, ActionPathFor};

use self::node::StateNode;

use super::space::NablaStateActionSpace;

pub mod node;
#[cfg(feature = "graphviz")]
pub mod graphviz;
pub mod search;

pub struct SearchTree<P> {
    positions: BTreeMap<P, NonZeroUsize>,
    nodes: Vec<StateNode>,
    in_neighborhoods: Vec<BTreeSet<usize>>,
}

#[derive(Debug)]
pub struct Transition2 {
    pub(crate) state_position: usize,
    pub(crate) action_position: usize,
}

impl<P> SearchTree<P> {
    pub fn new<Space: NablaStateActionSpace>(
        space: &Space,
        root: &Space::State,
        cost: &Space::Cost,
        h_theta: &[f32],
    ) -> Self {
        Self {
            positions: Default::default(),
            nodes: vec![StateNode::new(space, root, cost, h_theta)],
            in_neighborhoods: vec![BTreeSet::new()],
        }
    }

    pub fn sizes(&self) -> impl Iterator<Item = (usize, usize)> + '_
    where
        P: ActionPath,
    {
        let mut sizes = vec![(1, 0)];
        for (p, i) in self.positions.iter() {
            let len = p.len();
            if len >= sizes.len() {
                sizes.resize(len + 1, (0, 0));
            }
            sizes[len].0 += 1;
            if self.nodes[i.get()].is_exhausted() {
                sizes[len].1 += 1;
            }
        }
        sizes.into_iter()
    }

    pub fn roll_out_episodes<Space>(
        &mut self,
        space: &Space,
        root: &Space::State,
        state: &mut Space::State,
        cost: &mut Space::Cost,
        path: &mut P,
        transitions: &mut Vec<Transition2>,
        state_pos: &mut Option<NonZeroUsize>,
    )
    where
        Space: NablaStateActionSpace,
        Space::State: Clone,
        P: Ord + ActionPath + ActionPathFor<Space> + Clone,
    {
        loop {
            debug_assert_eq!(path.len(), transitions.len());
            debug_assert_eq!(self.positions.len() + 1, self.nodes.len());
            // for (p, pos) in self.positions.iter() {
            //     let p = p.actions_taken().collect::<Vec<_>>();
            //     println!("({pos}): {p:?}");
            // }
            // println!();
            let state_position = state_pos.map(NonZeroUsize::get).unwrap_or(0);
            let node = &mut self.nodes[state_position];
            let Some((action_position, action_data)) = node.next_action() else {
                if transitions.is_empty() {
                    debug_assert_eq!(*state_pos, None);
                    return;
                }
                debug_assert!(self.positions.contains_key(path));
                self.empty_transitions(
                    root,
                    state,
                    path,
                    transitions,
                    state_pos,
                );
                return self.roll_out_episodes(
                    space,
                    root,
                    state,
                    cost,
                    path,
                    transitions,
                    state_pos,
                );
            };
            transitions.push(Transition2 {
                state_position,
                action_position,
            });
            let action = action_data.action();
            unsafe { path.push_unchecked(action) };
            let action = space.action(action);
            space.act(state, &action);

            let next_position = action_data.next_position_mut();
            match next_position {
                Some(_) => {
                    debug_assert_eq!(self.positions.get(path).copied(), *next_position);
                    *state_pos = *next_position
                },
                None => match self.positions.get(path) {
                    Some(&pos) => {
                        let inserted = self.in_neighborhoods[pos.get()].insert(state_position);
                        debug_assert!(inserted);
                        *next_position = Some(pos);
                        *state_pos = Some(pos);
                    },
                    None => {
                        // we will insert a node at the end of the tree
                        let next_pos = (1 + self.positions.len()).try_into().unwrap();
                        *next_position = Some(next_pos);
                        *state_pos = *next_position;
                        self.in_neighborhoods.push(core::iter::once(state_position).collect());
                        self.positions.insert(path.clone(), next_pos);
                        *cost = space.cost(state);
                        if space.is_terminal(state) {
                            let c_star = space.evaluate(cost);
                            let node = StateNode::new_exhausted(c_star);
                            self.push_node(node);
                            self.empty_transitions(
                                root,
                                state,
                                path,
                                transitions,
                                state_pos,
                            );
                            return self.roll_out_episodes(
                                space,
                                root,
                                state,
                                cost,
                                path,
                                transitions,
                                state_pos,
                            );
                        }
                        return;
                    },
                },
            }
        }
    }

    pub(crate) fn push_node(&mut self, node: StateNode)
    where
        P: Ord + ActionPath,
    {
        self.nodes.push(node);
    }

    pub(crate) fn empty_transitions<Space>(
        &mut self,
        root: &Space::State,
        state: &mut Space::State,
        path: &mut P,
        transitions: &mut Vec<Transition2>,
        state_pos: &mut Option<NonZeroUsize>,
    )
    where
        Space: NablaStateActionSpace,
        Space::State: Clone,
        P: Ord + ActionPath + ActionPathFor<Space> + Clone,
    {
        debug_assert!(!transitions.is_empty());
        debug_assert!(self.positions.contains_key(path));
        let pos = state_pos.map(NonZeroUsize::get).unwrap_or(0);
        let mut c_star = self.nodes[pos].c_star;
        let mut exhausting = true;
        for transition in transitions.drain(..).rev() {
            let node = &mut self.nodes[transition.state_position];
            if exhausting {
                node.exhaust_action(transition.action_position);
                exhausting = node.is_exhausted();
            } else {
                node.update_c_stars(&mut c_star, transition.action_position);
            }
        }
        path.clear();
        state.clone_from(root);
        *state_pos = None;
    }

    pub(crate) fn write_observations<Space: NablaStateActionSpace>(
        &self,
        space: &Space,
        observations: &mut [f32],
        weights: &mut [f32],
    ) {
        let root_node = &self.nodes[0];
        let c_s = root_node.c;
        // dbg!(c_s);
        root_node.actions.iter().filter_map(|a| {
            a.next_position().map(|node_as| {
                let node_as = &self.nodes[node_as.get()];
                (a.action(), node_as.c, node_as.c_star)
            })
        }).for_each(|(a, c_as, c_as_star)| {
            let h = space.h_sa(c_s, c_as, c_as_star);
            // dbg!(h, a, c_as, c_as_star);
            observations[a] = h;
            weights[a] = 1.0;
        });
    }

    pub(crate) fn node_data(&self) -> Vec<(Option<&P>, f32, f32)> {
        core::iter::once((None, self.nodes[0].c, self.nodes[0].c_star))
            .chain(self.positions.iter().map(|(p, pos)| {
                let node = &self.nodes[pos.get()];
                (Some(p), node.c, node.c_star)
            }))
            .collect()
    }

    pub(crate) fn positions(&self) -> &BTreeMap<P, NonZeroUsize> {
        &self.positions
    }

    pub(crate) fn nodes(&self) -> &[StateNode] {
        &self.nodes
    }
}
