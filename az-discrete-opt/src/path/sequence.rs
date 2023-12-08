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

    fn push<Space>(&mut self, action: &Space::Action)
    where
        Space: StateActionSpace,
        Self: ActionPathFor<Space>,
    {
        let index = Space::index(action);
        unsafe { self.push_unchecked(index) }
    }

    fn actions_taken<Space>(&self) -> impl Iterator<Item = &'_ usize> + '_
    where
        Space: StateActionSpace,
        Self: ActionPathFor<Space>,
    {
        self.actions.iter()
    }
}

unsafe impl<Space: StateActionSpace> ActionPathFor<Space> for ActionSequence {}
