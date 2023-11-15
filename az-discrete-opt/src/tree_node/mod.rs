

// // use crate::{state::{State, Action}, path::ActionPathFor};

// // pub mod prohibitions;

use crate::{space::StateActionSpace, path::ActionPathFor};

pub trait TreeNode {
    type State;
    type Path;

    fn state(&self) -> &Self::State;
    fn path(&self) -> &Self::Path;
}

pub trait TreeNodeFor<Space>: TreeNode
where
    Self: TreeNode,
    Self::Path: ActionPathFor<Space>,
    Space: StateActionSpace,
{
    fn apply_action(&mut self, action: &Space::Action);
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
}