use crate::space::StateActionSpace;

use super::{ActionPath, ActionPathFor};

#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct ActionSequence {
    actions: Vec<usize>,
}

impl ActionPath for ActionSequence {
    fn new() -> Self {
        Self {
            actions: Vec::new(),
        }
    }

    fn len(&self) -> usize {
        self.actions.len()
    }

    unsafe fn push_unchecked(&mut self, action: usize) {
        self.actions.push(action)
    }

    fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    fn clear(&mut self) {
        self.actions.clear();
    }

    fn push<Space>(&mut self, space: &Space, action: &Space::Action)
    where
        Space: StateActionSpace,
        Self: ActionPathFor<Space>,
    {
        let index = space.index(action);
        unsafe { self.push_unchecked(index) }
    }

    fn actions_taken(&self) -> impl Iterator<Item = usize> + '_ {
        self.actions.iter().copied()
    }
}

unsafe impl<Space> ActionPathFor<Space> for ActionSequence {}
