pub mod axioms;
pub mod prohibited;
#[cfg(test)]
mod tests;

pub trait StateActionSpace {
    type State;
    type Action;
    const DIM: usize;
    fn index(action: &Self::Action) -> usize;
    fn from_index(index: usize) -> Self::Action;
    fn act(state: &mut Self::State, action: &Self::Action);
    fn follow(state: &mut Self::State, actions: impl Iterator<Item = Self::Action>) {
        for action in actions {
            Self::act(state, &action);
        }
    }
    // todo! iterator should depend on `state`
    // ??? fn actions<'a>(state: &'a Self::State) -> impl IntoIteratorIterator<Item = usize> + 'a;
    fn action_indices(state: &Self::State) -> impl Iterator<Item = usize>;
    fn is_terminal(state: &Self::State) -> bool {
        Self::action_indices(state).next().is_none()
    }
    fn has_action(state: &Self::State, action: &Self::Action) -> bool {
        let action_index = Self::index(action);
        Self::action_indices(state).any(|i| i == action_index)
    }
    fn write_vec(state: &Self::State, vector: &mut [f32]);
}
