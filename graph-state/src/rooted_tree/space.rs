use az_discrete_opt::space::StateActionSpace;

use super::{RootedOrderedTree, ordered_edge::OrderedEdge};

struct ConstrainedRootedOrderedTree<const N: usize>;

impl<const N: usize> StateActionSpace for ConstrainedRootedOrderedTree<N> {
    type State = RootedOrderedTree<N>;

    type Action = OrderedEdge;

    const DIM: usize = (N - 1)*(N - 2)/2 - 1;

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
        state.all_possible_parent_modifications().map(|a| Self::index(&a))
    }

    fn write_vec(state: &Self::State, vec: &mut [f32]) {
        todo!()
    }
}