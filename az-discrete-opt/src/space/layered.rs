use ringbuffer::RingBuffer;

use crate::state::layers::Layers;

use super::{StateActionSpace, axioms::{ActionOrderIndependent, ActionsNeverRepeat}};

pub struct Layered<const LAYERS: usize, Space> {
    _space: Space,
}

impl<const LAYERS: usize, Space> StateActionSpace for Layered<LAYERS, Space>
where
    Space: StateActionSpace,
    Space::State: Clone,
{
    type State = Layers<Space::State, LAYERS>;

    type Action = Space::Action;

    const DIM: usize = Space::DIM * LAYERS;

    fn index(action: &Self::Action) -> usize {
        Space::index(action)
    }

    fn from_index(index: usize) -> Self::Action {
        Space::from_index(index)
    }

    fn act(state: &mut Self::State, action: &Self::Action) {
        state.push_op(|s| {
            let mut s = s.clone();
            Space::act(&mut s, action);
            s
        })
    }

    fn action_indices(state: &Self::State) -> impl Iterator<Item = usize> {
        Space::action_indices(state.back())
    }

    fn write_vec(state: &Self::State, vector: &mut [f32]) {
        debug_assert_eq!(vector.len(), Self::DIM);
        vector.fill(0.0);
        vector.chunks_mut(Space::DIM)
            .zip(state.buffer().iter())
            .for_each(|(chunk, s)| Space::write_vec(s, chunk));
    }

    fn follow(state: &mut Self::State, actions: impl Iterator<Item = Self::Action>) {
        for action in actions {
            Self::act(state, &action);
        }
    }

    fn is_terminal(state: &Self::State) -> bool {
        Space::is_terminal(state.back())
    }

    fn has_action(state: &Self::State, action: &Self::Action) -> bool {
        Space::has_action(state.back(), action)
    }
}

unsafe impl<const LAYERS: usize, Space: ActionsNeverRepeat> ActionsNeverRepeat for Layered<LAYERS, Space> {}
unsafe impl<const LAYERS: usize, Space: ActionOrderIndependent> ActionOrderIndependent for Layered<LAYERS, Space> {}
