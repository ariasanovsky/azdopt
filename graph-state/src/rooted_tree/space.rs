use az_discrete_opt::{space::{StateActionSpace, axioms::{ActionOrderIndependent, ActionsNeverRepeat}}, nabla::space::NablaStateActionSpace};

use crate::simple_graph::edge::Edge;

use super::{ordered_edge::OrderedEdge, RootedOrderedTree, modify_parent_once::ROTWithParentPermissions};

pub struct ConstrainedRootedOrderedTree<const N: usize>;

impl<const N: usize> StateActionSpace for ConstrainedRootedOrderedTree<N> {
    type State = RootedOrderedTree<N>;

    type Action = OrderedEdge;

    const DIM: usize = (N - 1) * (N - 2) / 2 - 1;

    fn index(&self, action: &Self::Action) -> usize {
        action.index_ignoring_edge_0_1()
    }

    fn action(&self, index: usize) -> Self::Action {
        Self::Action::from_index_ignoring_edge_0_1(index)
    }

    fn act(&self, state: &mut Self::State, action: &Self::Action) {
        state.set_parent(action);
    }

    fn action_indices(&self, state: &Self::State) -> impl Iterator<Item = usize> {
        state
            .all_possible_parent_modifications_ignoring_last_vertex()
            .map(|a| self.index(&a))
    }

    fn write_vec(&self, state: &Self::State, vec: &mut [f32]) {
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
            let index = self.index(&edge);
            if index == 170 {
                println!("index = {index}, edge = {edge:?}, child = {child}, parent = {parent}");
            }
            vec[index] = 1.;
        }
    }
}

pub struct ROTModifyParentsOnce<const N: usize> {
    
}

impl<const N: usize> NablaStateActionSpace for ROTModifyParentsOnce<N> {
    type State = ROTWithParentPermissions<N>;

    type Action = OrderedEdge;

    type Reward = ();

    type Cost = f32;

    const STATE_DIM: usize = (N - 1) * (N - 2) / 2 - 1 + N - 3;

    const ACTION_DIM: usize = (N - 1) * (N - 2) / 2 - 1;

    fn action(&self, index: usize) -> Self::Action {
        todo!()
    }

    fn reward(&self, state: &Self::State, index: usize) -> Self::Reward {
        todo!()
    }

    fn act(&self, state: &mut Self::State, action: &Self::Action) {
        todo!()
    }

    fn action_data<'a>(
        &self,
        state: &'a Self::State,
    ) -> impl Iterator<Item = (usize, Self::Reward)> + 'a {
        let it = core::iter::empty();
        todo!();
        it
    }

    fn write_vec(&self, state: &Self::State, vector: &mut [f32]) {
        todo!()
    }

    fn cost(&self, state: &Self::State) -> Self::Cost {
        todo!()
    }

    fn evaluate(&self, cost: &Self::Cost) -> f32 {
        todo!()
    }

    fn g_theta_star_sa(&self, c_s: &Self::Cost, r_sa: Self::Reward, h_theta_sa: f32) -> f32 {
        todo!()
    }

    fn h_sa(&self, c_s: f32, c_as: f32, c_as_star: f32) -> f32 {
        todo!()
    }
}

unsafe impl<const N: usize> ActionOrderIndependent for ROTModifyParentsOnce<N> {}
unsafe impl<const N: usize> ActionsNeverRepeat for ROTModifyParentsOnce<N> {}