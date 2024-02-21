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

    unsafe fn undo_unchecked(&mut self, action: usize) {
        let removed = self.actions.remove(&action);
        debug_assert!(removed);
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

    fn extends_towards(&self, action: usize, target: &Self) -> bool {
        target.actions.contains(&action) && !self.actions.contains(&action)
    }
}
