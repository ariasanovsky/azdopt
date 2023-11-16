pub mod axioms;

pub trait StateActionSpace {
    type State;
    type Action;
    const DIM: usize;
    fn index(action: &Self::Action) -> usize;
    fn from_index(index: usize) -> Self::Action;
    fn act(state: &mut Self::State, action: &Self::Action);
    // todo! iterator should depend on `state`
    // ??? fn actions<'a>(state: &'a Self::State) -> impl IntoIteratorIterator<Item = usize> + 'a;
    fn actions(state: &Self::State) -> impl Iterator<Item = usize>;
    fn is_terminal(state: &Self::State) -> bool {
        Self::actions(state).next().is_none()
    }
    fn has_action(state: &Self::State, action: &Self::Action) -> bool {
        let action_index = Self::index(action);
        Self::actions(state).any(|i| i == action_index)
    }
    fn write_vec(state: &Self::State, vec: &mut [f32]);
}

pub trait ActionSpace: Sized {
    fn index<Space: StateActionSpace<Action = Self>>(&self) -> usize {
        Space::index(self)
    }
    fn from_index<Space: StateActionSpace<Action = Self>>(index: usize) -> Self {
        Space::from_index(index)
    }
}

// todo! `#[derive(ActionSpace)]` instead
impl<A> ActionSpace for A {}

pub trait StateSpace: Sized {
    fn act<Space: StateActionSpace<State = Self>>(&mut self, action: &Space::Action) {
        Space::act(self, action)
    }
    // todo! iterator should depend on `self`
    // ??? fn actions<Space>(&self) -> impl IntoIterator<Item = usize> + '_;
    fn actions<Space: StateActionSpace<State = Self>>(&self) -> impl Iterator<Item = usize> {
        Space::actions(self)
    }
    fn is_terminal<Space: StateActionSpace<State = Self>>(&self) -> bool {
        Space::is_terminal(self)
    }
    fn has_action<Space: StateActionSpace<State = Self>>(&self, action: &Space::Action) -> bool {
        Space::has_action(self, action)
    }
}

// todo! `#[derive(StateSpace)]` instead
impl<S> StateSpace for S {}

pub trait StateSpaceVec: Sized {
    fn write_vec<Space: StateActionSpace<State = Self>>(&self, vec: &mut [f32]) {
        debug_assert!(vec.len() == Space::DIM);
        Space::write_vec(self, vec)
    }
}

// todo! `#[derive(StateSpaceVec)]` instead
impl<S> StateSpaceVec for S {}
