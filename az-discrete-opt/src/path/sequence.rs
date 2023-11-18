use crate::space::StateActionSpace;

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
