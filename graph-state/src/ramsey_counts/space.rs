use az_discrete_opt::{nabla::space::NablaStateActionSpace, state::prohibit::WithProhibitions};

use crate::bitset::Bitset;

use super::{RamseyCounts, AssignColor, CountChange, TotalCounts};

pub struct RichRamseySpace<B, const N: usize, const E: usize> {
    // cost_fn: fn(&RamseyCounts<N, E, 2, B>) -> TotalCounts<2>,
    evaluate_fn: fn(&TotalCounts<2>) -> f32,
    _marker: core::marker::PhantomData<B>,
}

impl<B, const N: usize, const E: usize> RichRamseySpace<B, N, E> {
    pub fn new(
        // cost_fn: fn(&RamseyCounts<N, E, 2, B>) -> TotalCounts<2>,
        evaluate_fn: fn(&TotalCounts<2>) -> f32,
    ) -> Self {
        Self {
            // cost_fn,
            evaluate_fn,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<B: Bitset, const N: usize, const E: usize> NablaStateActionSpace for RichRamseySpace<B, N, E>
{
    type State = WithProhibitions<RamseyCounts<N, E, 2, B>>;

    type Action = AssignColor;

    type Reward = CountChange;

    type Cost = TotalCounts<2>;

    const STATE_DIM: usize = E * 6;

    const ACTION_DIM: usize = E * 2;

    // fn index(&self, action: &Self::Action) -> usize {
    //     todo!()
    // }

    // fn action(&self, index: usize) -> Self::Action {
    //     todo!()
    // }

    // fn act(&self, state: &mut Self::State, action: &Self::Action) {
    //     todo!()
    // }

    // fn action_indices(&self, state: &Self::State) -> impl Iterator<Item = usize> {
    //     todo!()
    // }

    fn write_vec(&self, state: &Self::State, vector: &mut [f32]) {
        debug_assert!(vector.len() == Self::STATE_DIM);
        vector.fill(0.);
        /* chunks are as follows:
        * 0/1: red/blue clique counts
        * 2/3: red/blue edge bools
        * 4/5: red/blue prohibited actions
        */
        let (clique_edge_vec, prohib_vec) = vector.split_at_mut(4 * E);
        let clique_counts = state.state.counts.iter().flat_map(|c| c.iter()).map(|c| *c as f32);
        let edge_bools = state.state.graph().graphs().iter().flat_map(|g| g.edge_bools()).map(|b| b.then_some(1.0f32).unwrap_or(0.));
        let clique_edge = clique_counts.chain(edge_bools);
        clique_edge_vec.iter_mut().zip(clique_edge).for_each(|(v, c)| *v = c);
        for a in state.prohibited_actions.iter() {
            prohib_vec[*a] = 1.;
        }
    }

    fn cost(&self, state: &Self::State) -> Self::Cost {
        state.state.clique_counts().clone()
    }

    fn evaluate(&self, cost: &Self::Cost) -> f32 {
        (self.evaluate_fn)(cost)
    }

    // fn c_theta_star_s_a(&self, c_s: f32, r_sa: Self::Reward, h_theta_s_a: f32) -> f32 {
    //     todo!()
    // }
}