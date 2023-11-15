use super::ActionPath;

#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct OrderedActionSet {
    pub(crate) actions: Vec<usize>,
}

impl ActionPath for OrderedActionSet {
    fn len(&self) -> usize {
        self.actions.len()
    }

    unsafe fn push_unchecked(&mut self, action: usize) {
        self.actions.push(action)
    }
}
