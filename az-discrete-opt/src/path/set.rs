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

    fn push<Space>(&mut self, action: &Space::Action)
    where
        Space: crate::space::StateActionSpace,
        Self: super::ActionPathFor<Space>,
    {
        let index = Space::index(action);
        unsafe { self.push_unchecked(index) }
    }
}
