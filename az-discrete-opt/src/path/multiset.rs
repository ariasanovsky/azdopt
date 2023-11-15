use std::collections::BTreeMap;

use super::ActionPath;

#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct ActionMultiset {
    pub(crate) actions: BTreeMap<usize, usize>,
}

impl ActionPath for ActionMultiset {
    fn len(&self) -> usize {
        self.actions.iter().map(|(_, count)| count).sum()
    }

    unsafe fn push_unchecked(&mut self, action: usize) {
        self.actions
            .entry(action)
            .and_modify(|count| *count += 1)
            .or_insert(1);
    }
}