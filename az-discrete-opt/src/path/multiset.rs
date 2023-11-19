use std::collections::BTreeMap;

use super::ActionPath;

#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct ActionMultiset {
    pub(crate) actions: BTreeMap<usize, usize>,
}

impl ActionPath for ActionMultiset {
    fn new() -> Self {
        Self {
            actions: BTreeMap::new(),
        }
    }

    fn len(&self) -> usize {
        self.actions.values().sum()
    }

    unsafe fn push_unchecked(&mut self, action: usize) {
        self.actions
            .entry(action)
            .and_modify(|count| *count += 1)
            .or_insert(1);
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
