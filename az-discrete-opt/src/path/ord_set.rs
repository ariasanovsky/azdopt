use super::ActionPath;

#[derive(Clone, Debug, Ord, PartialOrd, Eq, PartialEq)]
pub struct OrderedActionSet {
    pub(crate) actions: Vec<usize>,
}

impl ActionPath for OrderedActionSet {
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

    fn is_empty(&self) -> bool {
        self.actions.is_empty()
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
