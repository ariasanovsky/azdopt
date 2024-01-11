use az_discrete_opt::space::StateActionSpace;

use crate::{bitset::Bitset, simple_graph::bitset_graph::space::action::AddOrDeleteEdge};

use super::ConnectedBitsetGraph;

pub struct ConnectedAddOrDeleteEdge<const N: usize, B>(core::marker::PhantomData<B>);

impl<const N: usize, B> StateActionSpace for ConnectedAddOrDeleteEdge<N, B>
where
    B: Bitset + Clone + PartialEq,
    B::Bits: Clone,
{
    type State = ConnectedBitsetGraph<N, B>;

    type Action = AddOrDeleteEdge;

    const DIM: usize = N * (N - 1) / 2;

    fn index(&self, action: &Self::Action) -> usize {
        action.action_index::<N>()
    }

    fn action(&self, index: usize) -> Self::Action {
        Self::Action::from_action_index::<N>(index)
    }

    fn act(&self, state: &mut Self::State, action: &Self::Action) {
        match action {
            AddOrDeleteEdge::Add(e) => {
                debug_assert!(state.neighborhoods[e.max].contains(e.min as _).unwrap());
                debug_assert!(state.neighborhoods[e.min].contains(e.max as _).unwrap());
                unsafe { state.neighborhoods[e.max].add_or_remove_unchecked(e.min as _) };
                unsafe { state.neighborhoods[e.min].add_or_remove_unchecked(e.max as _) };
            }
            AddOrDeleteEdge::Delete(e) => {
                debug_assert!(!state.neighborhoods[e.max].contains(e.min as _).unwrap());
                debug_assert!(!state.neighborhoods[e.min].contains(e.max as _).unwrap());
                unsafe { state.neighborhoods[e.max].add_or_remove_unchecked(e.min as _) };
                unsafe { state.neighborhoods[e.min].add_or_remove_unchecked(e.max as _) };
            }
        }
    }

    fn action_indices(&self, state: &Self::State) -> impl Iterator<Item = usize> {
        let actions = state
            .action_kinds()
            .enumerate()
            .filter_map(|(i, b)| {
                b.map(|b| match b {
                    super::ActionKind::Add => i,
                    super::ActionKind::Delete => i + N * (N - 1) / 2,
                })
            })
            .collect::<Vec<_>>();
        actions.into_iter()
    }

    fn write_vec(&self, state: &Self::State, vec: &mut [f32]) {
        debug_assert!(vec.len() == Self::DIM);
        vec.iter_mut()
            .zip(state.edge_bools())
            .for_each(|(f, b)| match b {
                true => *f = 1.0,
                false => *f = 0.0,
            });
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        self.action_indices(state).next().is_none()
    }

    fn has_action(&self, state: &Self::State, action: &Self::Action) -> bool {
        let action_index = self.index(action);
        self.action_indices(state).any(|i| i == action_index)
    }
}


#[cfg(test)]
mod test {
    use az_discrete_opt::space::StateActionSpace;

    use super::*;
    use crate::{
        bitset::primitive::B32,
        simple_graph::{bitset_graph::BitsetGraph, edge::Edge},
    };

    #[test]
    fn c4_is_connected_and_has_no_cut_edges() {
        let graph: BitsetGraph<4, B32> = [(0, 1), (1, 2), (2, 3), (3, 0)]
            .as_ref()
            .try_into()
            .unwrap();
        let graph = graph.to_connected().unwrap();
        let len = graph
            .edges()
            .enumerate()
            .map(|(i, e)| {
                assert!(!graph.is_cut_edge(&e));
                i
            })
            .last()
            .unwrap()
            + 1;
        assert_eq!(len, 4);
    }

    #[test]
    fn p4_is_connected_and_all_edges_are_cut_edges() {
        let graph: BitsetGraph<4, B32> = [(0, 1), (1, 2), (2, 3)].as_ref().try_into().unwrap();
        let graph = graph.to_connected().unwrap();
        let len = graph
            .edges()
            .enumerate()
            .map(|(i, e)| {
                assert!(graph.is_cut_edge(&e));
                i
            })
            .last()
            .unwrap()
            + 1;
        assert_eq!(len, 3);
    }

    #[test]
    fn paw_is_connected_and_has_one_cut_edge() {
        let graph: BitsetGraph<4, B32> = [(0, 1), (0, 2), (1, 2), (2, 3)]
            .as_ref()
            .try_into()
            .unwrap();
        let graph = graph.to_connected().unwrap();
        let len = graph
            .edges()
            .enumerate()
            .map(|(i, e)| {
                if graph.is_cut_edge(&e) {
                    assert_eq!(e.vertices(), (3, 2));
                }
                i
            })
            .last()
            .unwrap()
            + 1;
        assert_eq!(len, 4);
    }

    #[test]
    fn complete_graph_on_four_vertices_has_all_possible_delete_actions() {
        let graph: ConnectedBitsetGraph<4, B32> = [(0, 1), (0, 2), (1, 2), (0, 3), (1, 3), (2, 3)]
            .as_ref()
            .try_into()
            .unwrap();
        let space = ConnectedAddOrDeleteEdge::<4, B32>(core::marker::PhantomData);
        let actions = space
            .action_indices(&graph)
            .map(|i| space.action(i))
            .collect::<Vec<_>>();
        assert_eq!(
            actions,
            vec![
                AddOrDeleteEdge::Delete(Edge::new(0, 1)),
                AddOrDeleteEdge::Delete(Edge::new(0, 2)),
                AddOrDeleteEdge::Delete(Edge::new(1, 2)),
                AddOrDeleteEdge::Delete(Edge::new(0, 3)),
                AddOrDeleteEdge::Delete(Edge::new(1, 3)),
                AddOrDeleteEdge::Delete(Edge::new(2, 3)),
            ]
        );
    }
}
