use ringbuffer::RingBuffer;

use crate::state::layers::Layers;

use super::{
    axioms::{ActionOrderIndependent, ActionsNeverRepeat},
    StateActionSpace,
};

pub struct Layered<const LAYERS: usize, Space> {
    pub space: Space,
}

impl<const LAYERS: usize, Space> Layered<LAYERS, Space> {
    pub const fn new(space: Space) -> Self {
        Self { space }
    }
}

impl<const LAYERS: usize, Space> StateActionSpace for Layered<LAYERS, Space>
where
    Space: StateActionSpace,
    Space::State: Clone,
{
    type State = Layers<Space::State, LAYERS>;

    type Action = Space::Action;

    const DIM: usize = Space::DIM * LAYERS;

    fn index(&self, action: &Self::Action) -> usize {
        self.space.index(action)
    }

    fn action(&self, index: usize) -> Self::Action {
        self.space.action(index)
    }

    fn act(&self, state: &mut Self::State, action: &Self::Action) {
        state.push_op(|s| {
            let mut s = s.clone();
            self.space.act(&mut s, action);
            s
        })
    }

    fn action_indices(&self, state: &Self::State) -> impl Iterator<Item = usize> {
        self.space.action_indices(state.back())
    }

    fn write_vec(&self, state: &Self::State, vector: &mut [f32]) {
        debug_assert_eq!(vector.len(), Self::DIM);
        vector.fill(0.0);
        vector
            .chunks_mut(Space::DIM)
            .zip(state.buffer().iter())
            .for_each(|(chunk, s)| self.space.write_vec(s, chunk));
    }

    fn follow(&self, state: &mut Self::State, actions: impl Iterator<Item = Self::Action>) {
        for action in actions {
            self.act(state, &action);
        }
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        self.space.is_terminal(state.back())
    }

    fn has_action(&self, state: &Self::State, action: &Self::Action) -> bool {
        self.space.has_action(state.back(), action)
    }
}

unsafe impl<const LAYERS: usize, Space: ActionsNeverRepeat> ActionsNeverRepeat
    for Layered<LAYERS, Space>
{
}
unsafe impl<const LAYERS: usize, Space: ActionOrderIndependent> ActionOrderIndependent
    for Layered<LAYERS, Space>
{
}
