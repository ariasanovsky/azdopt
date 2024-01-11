use az_discrete_opt::{space::{StateActionSpace, axioms::{ActionOrderIndependent, ActionsNeverRepeat}}, nabla::space::NablaStateActionSpace};

use crate::simple_graph::edge::Edge;

use super::{ordered_edge::OrderedEdge, RootedOrderedTree, modify_parent_once::ROTWithActionPermissions};

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

pub struct ROTModifyParentsOnce<const N: usize, C> {
    cost: fn(&RootedOrderedTree<N>) -> C,
    eval: fn(&C) -> f32,
}

impl<const N: usize, C> ROTModifyParentsOnce<N, C> {
    pub const fn new(
        cost: fn(&RootedOrderedTree<N>) -> C,
        eval: fn(&C) -> f32,
    ) -> Self {
        Self { cost, eval }
    }
}

impl<const N: usize, C> NablaStateActionSpace for ROTModifyParentsOnce<N, C> {
    type State = ROTWithActionPermissions<N>;

    type Action = OrderedEdge;

    type Reward = ();

    type Cost = C;

    const STATE_DIM: usize = (N - 1) * (N - 2) - 2;

    const ACTION_DIM: usize = (N - 1) * (N - 2) / 2 - 1;

    fn action(&self, index: usize) -> Self::Action {
        Self::Action::from_index_ignoring_edge_0_1(index)
    }

    fn reward(&self, _state: &Self::State, _index: usize) -> Self::Reward {
        ()
    }

    fn act(&self, state: &mut Self::State, action: &Self::Action) {
        let ROTWithActionPermissions { tree, permitted_actions } = state;
        debug_assert!(permitted_actions.contains(&action.index_ignoring_edge_0_1()));
        tree.set_parent(action);
        let child = action.child();
        for u in 0..child {
            let edge = Edge::new(u, child);
            let edge = OrderedEdge::new(edge);
            let index = edge.index_ignoring_edge_0_1();
            permitted_actions.remove(&index);
        }
    }

    fn action_data<'a>(
        &self,
        state: &'a Self::State,
    ) -> impl Iterator<Item = (usize, Self::Reward)> + 'a {
        let current_edge_positions = state.tree.edge_indices_ignoring_0_1_and_last_vertex().collect::<Vec<_>>();
        state.permitted_actions.iter().filter_map(move |action| {
            match current_edge_positions.contains(action) {
                true => None,
                false => Some((*action, ()))
            }
        })
    }

    fn write_vec(&self, state: &Self::State, vector: &mut [f32]) {
        debug_assert_eq!(vector.len(), Self::STATE_DIM);
        vector.fill(0.);
        let (current_parents_vec, permitted_actions_vec) = vector.split_at_mut(Self::ACTION_DIM);
        for edge_pos in state.tree.edge_indices_ignoring_0_1_and_last_vertex() {
            current_parents_vec[edge_pos] = 1.;
        }
        for &action in &state.permitted_actions {
            permitted_actions_vec[action] = 1.;
        }
        // println!("current_parents_vec   (total = {}) = {:?}", current_parents_vec.iter().sum::<f32>(), current_parents_vec);
        // println!("permitted_actions_vec (total = {}) = {:?}", permitted_actions_vec.iter().sum::<f32>(), permitted_actions_vec);
        // todo!()
    }

    fn cost(&self, state: &Self::State) -> Self::Cost {
        (self.cost)(&state.tree)
    }

    fn evaluate(&self, cost: &Self::Cost) -> f32 {
        (self.eval)(cost)
    }

    fn g_theta_star_sa(&self, c_s: &Self::Cost, _r_sa: Self::Reward, h_theta_sa: f32) -> f32 {
        h_theta_sa * self.evaluate(c_s)
    }

    fn h_sa(&self, c_s: f32, _c_as: f32, c_as_star: f32) -> f32 {
        c_as_star / c_s
    }
}

unsafe impl<const N: usize, C> ActionOrderIndependent for ROTModifyParentsOnce<N, C> {}
unsafe impl<const N: usize, C> ActionsNeverRepeat for ROTModifyParentsOnce<N, C> {}