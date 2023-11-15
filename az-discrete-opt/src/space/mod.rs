pub mod axioms;

pub trait StateActionSpace {
    type State;
    type Action;
    const DIM: usize;
    fn index(action: &Self::Action) -> usize;
    fn from_index(index: usize) -> Self::Action;
    fn act(state: &mut Self::State, action: &Self::Action);
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

pub trait Action: Sized {
    fn index<Space: StateActionSpace<Action = Self>>(&self) -> usize {
        Space::index(self)
    }
    fn from_index<Space: StateActionSpace<Action = Self>>(index: usize) -> Self {
        Space::from_index(index)
    }
}

impl<A> Action for A {}

pub trait State: Sized {
    fn act<Space: StateActionSpace<State = Self>>(&mut self, action: &Space::Action) {
        Space::act(self, action)
    }
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

impl<S> State for S {}

// pub trait StateVec<Space> {
//     const DIM: usize;
//     fn write_vec(&self, vec: &mut [f32]);
// }


// impl<Space, S> StateVec<Space> for S
// where
//     Space: StateActionSpace<State = S>,
// {
//     const DIM: usize = Space::DIM;
//     fn write_vec(&self, vec: &mut [f32]) {
//         Space::write_vec(self, vec)
//     }
// }
