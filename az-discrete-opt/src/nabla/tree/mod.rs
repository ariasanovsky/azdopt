use std::collections::BTreeMap;

use core::num::NonZeroUsize;

use crate::path::{ActionPath, ActionPathFor};

use self::node::{StateNode, Transition, SamplePattern, SearchPolicy};

use super::space::NablaStateActionSpace;

pub mod node;
pub mod update_nodes;

pub struct SearchTree<P> {
    positions: BTreeMap<P, NonZeroUsize>,
    nodes: Vec<StateNode2>,
}

pub struct StateNode2;

pub struct Transition2;

pub struct SearchResult;

impl<P> SearchTree<P> {
    pub fn new<Space: NablaStateActionSpace>(
        space: &Space,
        root: &Space::State,
        cost: &Space::Cost,
        h_theta: &[f32],
        root_action_pattern: SamplePattern,
    ) -> Self {
        Self {
            positions: Default::default(),
            nodes: vec![StateNode2],
        }
    }

    pub fn sizes(&self) -> impl Iterator<Item = usize> + '_ {
        let it = core::iter::empty();
        todo!();
        it
    }

    pub fn roll_out_episodes<Space>(
        &mut self,
        space: &Space,
        root: &Space::State,
        state: &mut Space::State,
        cost: &mut Space::Cost,
        path: &mut P,
        transitions: &mut Vec<Transition2>,
        policy: impl Fn(usize) -> SearchPolicy,
    ) -> SearchResult
    where
        Space: NablaStateActionSpace,
        P: Ord + ActionPath + ActionPathFor<Space>,
    {
        let Self { positions, nodes } = self;
        todo!();
        // let transition = root_node.next_transition(policy(0)).ok()?;
        // let a = transition.action_index();
        // let action = space.action(a);
        // space.act(state, &action);
        // unsafe { path.push_unchecked(a) };
        // let mut transitions = vec![transition];

        // for (i, level) in levels.iter_mut().enumerate() {
        //     // I hate Polonius case III
        //     let node = match level.contains_key(path) {
        //         true => level.get_mut(path).unwrap(),
        //         false => return Some((transitions, NodeKind::New(level)),)
        //     };
        //     match node.next_transition(policy(i+1)) {
        //         Ok(trans) => {
        //             let a = trans.action_index();
        //             let action = space.action(a);
        //             space.act(state, &action);
        //             unsafe { path.push_unchecked(a) };
        //             transitions.push(trans)
        //         },
        //         Err(c) => return Some((transitions, NodeKind::OldExhausted { c_s_t_theta_star: c }))
        //     }
        // }
        // Some((transitions, NodeKind::NewLevel))
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

    // #[cfg(feature = "rayon")]
    // pub(crate) fn _par_next_roots(&self) -> impl rayon::iter::ParallelIterator<Item = (Option<&P>, usize, f32)> + '_
    // where
    //     P: Ord + Sync,
    // {
    //     use rayon::iter::{ParallelIterator, IntoParallelRefIterator};

    //     let next_from_roots =
    //         self.root_node._par_next_roots()
    //         .map(|(a, c_star)| (None, a, c_star));
    //     let next_from_nodes = self.levels.par_iter().flat_map(|level| {
    //         level.par_iter().flat_map(|(p, n)| {
    //             n._par_next_roots().map(move |(a, c_star)| (Some(p), a, c_star))
    //         })
    //     });
    //     next_from_roots.chain(next_from_nodes)
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
