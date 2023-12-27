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
    fn action(&self, index: usize) -> Self::Action;
    fn reward(&self, state: &Self::State, index: usize) -> Self::Reward;
    fn act(&self, state: &mut Self::State, action: &Self::Action);
    // fn follow(&self, state: &mut Self::State, actions: impl Iterator<Item = Self::Action>) {
    //     for action in actions {
    //         self.act(state, &action);
    //     }
    // }
    fn action_data<'a>(&self, state: &'a Self::State) -> impl Iterator<Item = (usize, Self::Reward)> + 'a;
    fn is_terminal(&self, state: &Self::State) -> bool {
        self.action_data(state).next().is_none()
    }
    // fn has_action(&self, state: &Self::State, action: &Self::Action) -> bool {
    //     let action_index = self.index(action);
    //     self.action_indices(state).any(|i| i == action_index)
    // }
    fn write_vec(&self, state: &Self::State, vector: &mut [f32]);
    fn cost(&self, state: &Self::State) -> Self::Cost;
    fn evaluate(&self, cost: &Self::Cost) -> f32;
    fn g_theta_star_sa(&self, c_s: &Self::Cost, r_sa: Self::Reward, h_theta_sa: f32) -> f32;
    fn h_sa(&self, c_s: &Self::Cost, r_sa: Self::Reward, g_sa: f32) -> f32;
}

impl<const L: usize, Space: NablaStateActionSpace> NablaStateActionSpace for Layered<L, Space>
where
    Space::State: Clone,
{
    type State = Layers<Space::State, L>;

    type Action = Space::Action;

    type Reward = Space::Reward;

    type Cost = Space::Cost;

    const STATE_DIM: usize = Space::STATE_DIM * L;

    const ACTION_DIM: usize = Space::ACTION_DIM;

    fn action(&self, index: usize) -> Self::Action {
        self.space.action(index)
    }

    fn reward(&self, state: &Self::State, index: usize) -> Self::Reward {
        self.space.reward(state.back(), index)
    }

    fn act(&self, state: &mut Self::State, action: &Self::Action) {
        state.push_op(|s| {
            let mut s = s.clone();
            self.space.act(&mut s, action);
            s
        });
    }

    fn action_data<'a>(&self, state: &'a Self::State) -> impl Iterator<Item = (usize, Self::Reward)> + 'a {
        self.space.action_data(state.back())
    }

    fn write_vec(&self, state: &Self::State, vector: &mut [f32]) {
        debug_assert!(
            vector.len() == Self::STATE_DIM,
            "{} != {}",
            vector.len(),
            Self::STATE_DIM,
        );
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

    fn g_theta_star_sa(&self, c_s: &Self::Cost, r_sa: Self::Reward, h_theta_sa: f32) -> f32 {
        self.space.g_theta_star_sa(c_s, r_sa, h_theta_sa)
    }

    fn h_sa(&self, c_s: &Self::Cost, r_sa: Self::Reward, g_sa: f32) -> f32 {
        self.space.h_sa(c_s, r_sa, g_sa)
    }
}
