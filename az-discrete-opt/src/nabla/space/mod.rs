use ringbuffer::RingBuffer;

use crate::{space::layered::Layered, state::layers::Layers};

pub trait NablaStateActionSpace {
    type State;
    type Action;
    type Reward;
    type Cost;

    const STATE_DIM: usize;
    const ACTION_DIM: usize;

    // fn index(&self, action: &Self::Action) -> usize;
    // fn action(&self, index: usize) -> Self::Action;
    // fn act(&self, state: &mut Self::State, action: &Self::Action);
    // fn follow(&self, state: &mut Self::State, actions: impl Iterator<Item = Self::Action>) {
    //     for action in actions {
    //         self.act(state, &action);
    //     }
    // }
    // fn action_indices(&self, state: &Self::State) -> impl Iterator<Item = usize>;
    // fn is_terminal(&self, state: &Self::State) -> bool {
    //     self.action_indices(state).next().is_none()
    // }
    // fn has_action(&self, state: &Self::State, action: &Self::Action) -> bool {
    //     let action_index = self.index(action);
    //     self.action_indices(state).any(|i| i == action_index)
    // }
    fn write_vec(&self, state: &Self::State, vector: &mut [f32]);
    fn cost(&self, state: &Self::State) -> Self::Cost;
    fn evaluate(&self, cost: &Self::Cost) -> f32;
    // fn c_theta_star_s_a(&self, c_s: f32, r_sa: Self::Reward, h_theta_s_a: f32) -> f32;
}

impl<const L: usize, Space: NablaStateActionSpace> NablaStateActionSpace for Layered<L, Space> {
    type State = Layers<Space::State, L>;

    type Action = Space::Action;

    type Reward = Space::Reward;

    type Cost = Space::Cost;

    const STATE_DIM: usize = Space::STATE_DIM * L;

    const ACTION_DIM: usize = Space::ACTION_DIM;

    fn write_vec(&self, state: &Self::State, vector: &mut [f32]) {
        debug_assert!(vector.len() == Self::STATE_DIM * L);
        state.buffer().iter().zip(vector.chunks_exact_mut(Space::STATE_DIM)).for_each(|(s, s_host)| {
            self.space.write_vec(s, s_host);
        });
    }

    fn cost(&self, state: &Self::State) -> Self::Cost {
        self.space.cost(state.back())
    }

    fn evaluate(&self, cost: &Self::Cost) -> f32 {
        self.space.evaluate(cost)
    }
}