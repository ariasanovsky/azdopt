use std::collections::{BTreeMap, BTreeSet};

use core::num::NonZeroUsize;

use crate::path::{ActionPath, ActionPathFor};

use self::node::{StateNode2, StateNode};

use super::space::NablaStateActionSpace;

pub mod node;
#[cfg(feature = "graphviz")]
pub mod graphviz;
pub mod search;
pub mod update_nodes;

pub struct SearchTree<P> {
    positions: BTreeMap<P, NonZeroUsize>,
    nodes: Vec<StateNode2>,
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
            nodes: vec![StateNode2::new(space, root, cost, h_theta)],
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
                            let node = StateNode2::new_exhausted(c_star);
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

    // pub(crate) fn insert_new_node(
    //     &mut self,
    //     path: P,
    //     node: StateNode2,
    // )
    // where
    //     P: Ord + ActionPath,
    // {
    //     let Self {
    //         root_node: _,
    //         levels,
    //     } = self;
    //     debug_assert_eq!(path.len(), levels.len() + 1);
    //     let level = BTreeMap::from_iter(core::iter::once((path, node)));
    //     levels.push(level);
    // }

    // pub(crate) fn root_node(&self) -> &StateNode {
    //     &self.root_node
    // }

    #[cfg(feature = "rayon")]
    pub(crate) fn par_nodes(&self) -> impl rayon::iter::ParallelIterator<Item = (Option<&P>, f32)> + '_
    where
        P: Ord + Sync,
    {
        let it = rayon::iter::empty();
        // use rayon::iter::{ParallelIterator, IntoParallelRefIterator};

        // rayon::iter::once((None, self.root_node.cost()))
        //     .chain(self.levels.par_iter().flat_map(|level| {
        //         level.par_iter().map(|(p, n)| (Some(p), n.cost()))
        //     }))
        todo!();
        it
    }

    pub(crate) fn push_node(&mut self, node: StateNode2)
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
            println!("transition: {:?}", transition);
            println!("c_star: {:?}", c_star);
            println!("exhausting: {:?}", exhausting);
            println!();
            let node = &mut self.nodes[transition.state_position];
            if exhausting {
                node.exhaust_action(transition.action_position);
                exhausting = node.is_exhausted();
            } else {
                println!("not exhausting");
                node.update_c_stars(&mut c_star, transition.action_position);
            }
        }
        path.clear();
        state.clone_from(root);
        *state_pos = None;
    }

    // pub(crate) fn reset_search(
    //     &mut self,
    //     path: &mut P,
    //     transitions: &mut Vec<Transition2>,
    //     last_pos: usize,
    // )
    // where
    //     P: ActionPath,
    // {
    //     let Self {
    //         positions: _,
    //         nodes,
    //         in_neighborhoods,
    //     } = self;
    //     let mut previous_parents: VecDeque<usize> = Default::default();
    //     let mut pos = last_pos;
    //     let mut c_star = nodes[pos].c_star;
    //     let path_nodes = transitions.iter().map(|t| t.state_position).collect::<BTreeSet<_>>();
    //     transitions.drain(..).rev().for_each(|t| {
    //         let Transition2 { state_position, action_position } = t;
    //         if nodes[pos].is_exhausted() {
    //             nodes[state_position].actions[action_position].g_sa = None
    //         } else {
    //             let c_star_below = &mut nodes[state_position].c_star;
    //             if *c_star_below > c_star {
    //                 *c_star_below = c_star;
    //                 // todo!()
    //             } else {
    //                 let g_sa = &mut nodes[state_position].actions[action_position].g_sa.unwrap();
    //                 *g_sa -= 0.5;
    //                 // todo!()
    //             }
    //         }
            
    //         previous_parents.extend(
    //             in_neighborhoods[pos]
    //             .iter()
    //             .filter(|&&p| !path_nodes.contains(&p))
    //         );
    //         while let Some(p) = previous_parents.pop_front() {
    //             let c_star_p = &mut nodes[p].c_star;
    //             if *c_star_p > c_star {
    //                 *c_star_p = c_star;
    //                 previous_parents.extend(
    //                     in_neighborhoods[p].iter()
    //                     .filter(|&&p| !path_nodes.contains(&p))
    //                 );
    //             }
    //         }
            
    //         pos = state_position;
    //         c_star = nodes[pos].c_star;
    //     });
    //     // todo!();
    //     path.clear();
    // }
}

pub enum NodeKind<'roll_out, P> {
    NewLevel,
    New(&'roll_out mut BTreeMap<P, StateNode>),
    OldExhausted { c_s_t_theta_star: f32 },
}

impl<P> NodeKind<'_, P> {
    pub fn is_new(&self) -> bool {
        !matches!(self, Self::OldExhausted { .. })
    }
}
