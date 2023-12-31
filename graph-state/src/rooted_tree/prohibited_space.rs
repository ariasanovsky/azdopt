use az_discrete_opt::{
    space::{
        axioms::{ActionOrderIndependent, ActionsNeverRepeat},
        StateActionSpace,
    },
    state::prohibit::WithProhibitions,
};

use crate::simple_graph::edge::Edge;

use super::{ordered_edge::OrderedEdge, space::ConstrainedRootedOrderedTree};

pub struct ProhibitedConstrainedRootedOrderedTree<const N: usize>;

impl<const N: usize> StateActionSpace for ProhibitedConstrainedRootedOrderedTree<N> {
    type State = WithProhibitions<<ConstrainedRootedOrderedTree<N> as StateActionSpace>::State>;

    type Action = <ConstrainedRootedOrderedTree<N> as StateActionSpace>::Action;

    const DIM: usize =
        (N - 1) * (N - 2) / 2 - 1 + <ConstrainedRootedOrderedTree<N> as StateActionSpace>::DIM;

    fn index(&self, action: &Self::Action) -> usize {
        ConstrainedRootedOrderedTree::<N>.index(action)
    }

    fn action(&self, index: usize) -> Self::Action {
        ConstrainedRootedOrderedTree::<N>.action(index)
    }

    fn act(&self, state: &mut Self::State, action: &Self::Action) {
        debug_assert!(
            !state.prohibited_actions.contains(&170),
            "state = {state:?}",
        );
        let WithProhibitions {
            state,
            prohibited_actions,
        } = state;
        // act on state
        ConstrainedRootedOrderedTree::<N>.act(state, action);
        // prohibit modifying the parent of the child
        let (child, parent) = action.child_parent();
        debug_assert_ne!(child, 0);
        debug_assert_ne!(child, N - 1);
        let new_prohibited_actions = (0..parent)
            .chain(parent + 1..child)
            .map(|i| OrderedEdge::new(Edge::new(i, child)))
            .map(|a| self.index(&a));
        prohibited_actions.extend(new_prohibited_actions);
        debug_assert!(!prohibited_actions.contains(&170), "state = {state:?}",);
    }

    fn action_indices(&self, state: &Self::State) -> impl Iterator<Item = usize> {
        let WithProhibitions {
            state,
            prohibited_actions,
        } = state;
        let raw_indices = ConstrainedRootedOrderedTree::<N>.action_indices(state);
        raw_indices.filter(move |&i| !prohibited_actions.contains(&i))
    }

    fn write_vec(&self, state: &Self::State, vec: &mut [f32]) {
        debug_assert_eq!(vec.len(), Self::DIM);
        let (state_vec, prohib_vec) =
            vec.split_at_mut(<ConstrainedRootedOrderedTree<N> as StateActionSpace>::DIM);
        let WithProhibitions {
            state,
            prohibited_actions,
        } = state;
        ConstrainedRootedOrderedTree::<N>.write_vec(state, state_vec);
        prohib_vec.fill(0.);
        for &i in prohibited_actions {
            prohib_vec[i] = 1.;
        }
    }
}

unsafe impl<const N: usize> ActionOrderIndependent for ProhibitedConstrainedRootedOrderedTree<N> {}
unsafe impl<const N: usize> ActionsNeverRepeat for ProhibitedConstrainedRootedOrderedTree<N> {}
