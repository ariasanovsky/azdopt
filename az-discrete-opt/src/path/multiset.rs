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

    unsafe fn undo_unchecked(&mut self, action: usize) {
        let count = self.actions.get_mut(&action).unwrap();
        if *count == 1 {
            self.actions.remove(&action).unwrap();
        } else {
            *count -= 1;
        }
    }

    fn is_empty(&self) -> bool {
        self.actions.is_empty()
    }

    fn clear(&mut self) {
        self.actions.clear();
    }

    fn actions_taken(&self) -> impl Iterator<Item = usize> + '_ {
        self.actions
            .iter()
            .flat_map(|(action, count)| (0..*count).map(move |_| action))
            .copied()
    }

    fn extends_towards(&self, action: usize, target: &Self) -> bool {
        todo!()
    }
}
