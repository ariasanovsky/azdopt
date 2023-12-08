use az_discrete_opt::space::StateActionSpace;

use crate::simple_graph::edge::Edge;

use super::{ordered_edge::OrderedEdge, RootedOrderedTree};

pub struct ConstrainedRootedOrderedTree<const N: usize>;

impl<const N: usize> StateActionSpace for ConstrainedRootedOrderedTree<N> {
    type State = RootedOrderedTree<N>;

    type Action = OrderedEdge;

    const DIM: usize = (N - 1) * (N - 2) / 2 - 1;

    fn index(action: &Self::Action) -> usize {
        action.index_ignoring_edge_0_1()
    }

    fn from_index(index: usize) -> Self::Action {
        Self::Action::from_index_ignoring_edge_0_1(index)
    }

    fn act(state: &mut Self::State, action: &Self::Action) {
        state.set_parent(action);
    }

    fn action_indices(state: &Self::State) -> impl Iterator<Item = usize> {
        state
            .all_possible_parent_modifications_ignoring_last_vertex()
            .map(|a| Self::index(&a))
    }

    fn write_vec(state: &Self::State, vec: &mut [f32]) {
        debug_assert_eq!(vec.len(), Self::DIM);
        vec.fill(0.);
        for (child, parent) in state
            .parents_ignoring_last_vertex()
            .iter()
            .enumerate()
            .skip(2)
        {
            let edge = Edge::new(*parent, child);
            let edge = OrderedEdge::new(edge);
            let index = Self::index(&edge);
            if index == 170 {
                println!("index = {index}, edge = {edge:?}, child = {child}, parent = {parent}");
            }
            vec[index] = 1.;
        }
    }
}
