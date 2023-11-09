use az_discrete_opt::state::{ProhibitsActions, State, StateVec};
use itertools::Itertools;

use crate::simple_graph::{bitset_graph::state::Action, edge::Edge};

use super::ConnectedBitsetGraph;

impl<const N: usize> az_discrete_opt::state::Action<ConnectedBitsetGraph<N>> for Action {
    fn index(&self) -> usize {
        <Self as az_discrete_opt::state::Action<
            crate::simple_graph::bitset_graph::BitsetGraph<N>,
        >>::index(self)
    }

    unsafe fn from_index_unchecked(index: usize) -> Self {
        let e = N * (N - 1) / 2;
        if index < e {
            Self::Add(Edge::from_colex_position(index))
        } else {
            Self::Delete(Edge::from_colex_position(index - e))
        }
    }
}

impl<const N: usize> State for ConnectedBitsetGraph<N> {
    type Actions = Action;

    fn actions(&self) -> impl Iterator<Item = Self::Actions> {
        let Self { neighborhoods } = self;
        neighborhoods.iter().enumerate().flat_map(move |(v, n)| {
            (0..v).filter_map(move |u| {
                let e = unsafe { Edge::new_unchecked(v, u) };
                if n.contains(u) {
                    if self.is_cut_edge(&e) {
                        None
                    } else {
                        Some(Action::Delete(e))
                    }
                } else {
                    Some(Action::Add(e))
                }
            })
        })
    }

    unsafe fn act_unchecked(&mut self, action: &Self::Actions) {
        let Self { neighborhoods } = self;
        match action {
            Action::Add(e) | Action::Delete(e) => {
                let (v, u) = e.vertices();
                neighborhoods[u].add_or_remove_unchecked(v);
                neighborhoods[v].add_or_remove_unchecked(u);
            }
        }
    }
}

impl<const N: usize> StateVec for ConnectedBitsetGraph<N> {
    const STATE_DIM: usize = N * (N - 1) / 2;

    const AVAILABLE_ACTIONS_BOOL_DIM: usize = N * (N - 1);

    fn write_vec_state_dims(&self, state_vec: &mut [f32]) {
        self.edge_bools().zip_eq(state_vec).for_each(|(b, f)| {
            if b {
                *f = 1.;
            } else {
                *f = 0.;
            }
        });
    }

    fn write_vec_actions_dims(&self, action_vec: &mut [f32]) {
        let (adds, deletes) = action_vec.split_at_mut(N * (N - 1) / 2);
        self.action_kinds()
            .zip_eq(adds)
            .zip_eq(deletes)
            .for_each(|((b, add), delete)| {
                use crate::simple_graph::connected_bitset_graph::ActionKind;
                (*add, *delete) = match b {
                    Some(ActionKind::Add) => (1., 0.),
                    Some(ActionKind::Delete) => (0., 1.),
                    None => (0., 0.),
                }
            });
    }
}

impl<const N: usize> ProhibitsActions for ConnectedBitsetGraph<N> {
    type Action = Action;

    unsafe fn update_prohibited_actions_unchecked(
        &self,
        actions: &mut std::collections::BTreeSet<usize>,
        action: &Self::Action,
    ) {
        match action {
            Action::Add(e) | Action::Delete(e) => {
                actions.insert(e.colex_position());
                actions.insert(e.colex_position() + N * (N - 1) / 2);
            }
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::simple_graph::bitset_graph::BitsetGraph;

    #[test]
    fn c4_is_connected_and_has_no_cut_edges() {
        let graph: BitsetGraph<4> = [(0, 1), (1, 2), (2, 3), (3, 0)]
            .as_ref()
            .try_into()
            .unwrap();
        let graph = graph.to_connected().unwrap();
        let len = graph
            .edges()
            .enumerate()
            .map(|(i, e)| {
                debug_assert!(!graph.is_cut_edge(&e));
                i
            })
            .last()
            .unwrap()
            + 1;
        debug_assert_eq!(len, 4);
    }

    #[test]
    fn p4_is_connected_and_all_edges_are_cut_edges() {
        let graph: BitsetGraph<4> = [(0, 1), (1, 2), (2, 3)].as_ref().try_into().unwrap();
        let graph = graph.to_connected().unwrap();
        let len = graph
            .edges()
            .enumerate()
            .map(|(i, e)| {
                debug_assert!(graph.is_cut_edge(&e));
                i
            })
            .last()
            .unwrap()
            + 1;
        debug_assert_eq!(len, 3);
    }

    #[test]
    fn paw_is_connected_and_has_one_cut_edge() {
        let graph: BitsetGraph<4> = [(0, 1), (0, 2), (1, 2), (2, 3)]
            .as_ref()
            .try_into()
            .unwrap();
        let graph = graph.to_connected().unwrap();
        let len = graph
            .edges()
            .enumerate()
            .map(|(i, e)| {
                if graph.is_cut_edge(&e) {
                    debug_assert_eq!(e.vertices(), (3, 2));
                }
                i
            })
            .last()
            .unwrap()
            + 1;
        debug_assert_eq!(len, 4);
    }

    #[test]
    fn complete_graph_on_four_vertices_has_all_possible_delete_actions() {
        let graph: BitsetGraph<4> = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
            .as_ref()
            .try_into()
            .unwrap();
        let graph = graph.to_connected().unwrap();
        let actions = graph.actions().collect::<Vec<_>>();
        debug_assert_eq!(
            actions,
            vec![
                Action::Delete(Edge::new(0, 1)),
                Action::Delete(Edge::new(0, 2)),
                Action::Delete(Edge::new(1, 2)),
                Action::Delete(Edge::new(0, 3)),
                Action::Delete(Edge::new(1, 3)),
                Action::Delete(Edge::new(2, 3)),
            ]
        );
    }
}
