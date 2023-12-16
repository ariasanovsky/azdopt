pub mod axioms;
pub mod prohibited;
#[cfg(feature = "layers")]
pub mod layered;
#[cfg(test)]
mod tests;

pub trait StateActionSpace {
    type State;
    type Action;
    const DIM: usize;
    fn index(&self, action: &Self::Action) -> usize;
    fn action(&self, index: usize) -> Self::Action;
    fn act(&self, state: &mut Self::State, action: &Self::Action);
    fn follow(&self, state: &mut Self::State, actions: impl Iterator<Item = Self::Action>) {
        for action in actions {
            self.act(state, &action);
        }
    }
    // todo! iterator should depend on `state`
    // ??? fn actions<'a>(state: &'a Self::State) -> impl IntoIteratorIterator<Item = usize> + 'a;
    fn action_indices(&self, state: &Self::State) -> impl Iterator<Item = usize>;
    fn is_terminal(&self, state: &Self::State) -> bool {
        self.action_indices(state).next().is_none()
    }
    fn has_action(&self, state: &Self::State, action: &Self::Action) -> bool {
        let action_index = self.index(action);
        self.action_indices(state).any(|i| i == action_index)
    }
    fn write_vec(&self, state: &Self::State, vector: &mut [f32]);
}
