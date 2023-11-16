use std::collections::BTreeSet;

use super::ActionPath;

#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct ActionSet {
    pub(crate) actions: BTreeSet<usize>,
}

impl ActionPath for ActionSet {
    fn new() -> Self {
        Self { actions: BTreeSet::new() }
    }

    fn len(&self) -> usize {
        self.actions.len()
    }

    unsafe fn push_unchecked(&mut self, action: usize) {
        let inserted = self.actions.insert(action);
        debug_assert!(inserted);
    }
}
