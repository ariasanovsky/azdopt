use crate::{space::StateActionSpace, tree_node::{TreeNodeFor, MutRefNode}};

use super::{ActionPath, ActionPathFor};

#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct ActionSequence {
    actions: Vec<usize>,
}

impl ActionPath for ActionSequence {
    fn new() -> Self {
        Self { actions: Vec::new() }
    }

    fn len(&self) -> usize {
        self.actions.len()
    }

    unsafe fn push_unchecked(&mut self, action: usize) {
        self.actions.push(action)
    }
}

unsafe impl<Space: StateActionSpace> ActionPathFor<Space> for ActionSequence {}

impl<'a, Space: StateActionSpace, P> TreeNodeFor<Space> for MutRefNode<'a, Space::State, P>
where
    P: ActionPathFor<Space>,
{
    fn apply_action(&mut self, action: &<Space as StateActionSpace>::Action) {
        // todo!("pass in a Fn that updates a");
        let Self { state, path } = self;
        let index = Space::index(action);
        unsafe { path.push_unchecked(index) }
        Space::act(state, action)
    }
}
