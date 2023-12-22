use az_discrete_opt::{nabla::space::NablaStateActionSpace, state::prohibit::WithProhibitions};

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

impl<B, const N: usize, const E: usize> NablaStateActionSpace for RichRamseySpace<B, N, E>
{
    type State = WithProhibitions<RamseyCounts<N, E, 2, B>>;

    type Action = AssignColor;

    type Reward = CountChange;

    type Cost = TotalCounts<2>;

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

    // fn write_vec(&self, state: &Self::State, vector: &mut [f32]) {
    //     todo!()
    // }

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