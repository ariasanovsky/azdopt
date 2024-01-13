use std::collections::BTreeSet;

use super::ActionPath;

#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct ActionSet {
    pub(crate) actions: BTreeSet<usize>,
}

impl ActionPath for ActionSet {
    fn new() -> Self {
        Self {
            actions: BTreeSet::new(),
        }
    }

    fn len(&self) -> usize {
        self.actions.len()
    }

    unsafe fn push_unchecked(&mut self, action: usize) {
        let inserted = self.actions.insert(action);
        debug_assert!(inserted);
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
