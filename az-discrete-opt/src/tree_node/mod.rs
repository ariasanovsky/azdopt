use crate::{path::ActionPathFor, space::StateActionSpace};

pub trait TreeNode {
    type State;
    type Path;

    fn state(&self) -> &Self::State;
    fn path(&self) -> &Self::Path;
    fn apply_action<Space>(&mut self, action: &Space::Action)
    where
        Space: StateActionSpace<State = Self::State>,
        Self::Path: ActionPathFor<Space>;
}

pub struct MutRefNode<'a, S, P> {
    pub(crate) state: &'a mut S,
    pub(crate) path: &'a mut P,
}

impl<'a, S, P> MutRefNode<'a, S, P> {
    pub fn new(state: &'a mut S, path: &'a mut P) -> Self {
        Self { state, path }
    }
}

impl<'a, S, P> TreeNode for MutRefNode<'a, S, P> {
    type State = S;

    type Path = P;

    fn state(&self) -> &Self::State {
        self.state
    }

    fn path(&self) -> &Self::Path {
        self.path
    }

    fn apply_action<Space>(&mut self, action: &Space::Action)
    where
        Space: StateActionSpace<State = Self::State>,
        Self::Path: ActionPathFor<Space>,
    {
        let Self { state, path } = self;
        let index = Space::index(action);
        unsafe { path.push_unchecked(index) }
        Space::act(state, action)
    }
}
