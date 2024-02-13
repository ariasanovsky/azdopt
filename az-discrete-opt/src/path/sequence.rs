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

    unsafe fn undo_unchecked(&mut self, action: usize) {
        let last = self.actions.pop().unwrap();
        debug_assert_eq!(last, action);
    }

    fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    fn clear(&mut self) {
        self.actions.clear();
    }

    fn actions_taken(&self) -> impl Iterator<Item = usize> + '_ {
        self.actions.iter().copied()
    }
}

unsafe impl<Space> ActionPathFor<Space> for ActionSequence {}
