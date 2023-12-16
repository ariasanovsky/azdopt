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

    fn clear(&mut self) {
        self.actions.clear();
    }

    fn push<Space>(&mut self, space: &Space, action: &Space::Action)
    where
        Space: crate::space::StateActionSpace,
        Self: super::ActionPathFor<Space>,
    {
        let index = space.index(action);
        unsafe { self.push_unchecked(index) }
    }

    fn actions_taken<Space>(&self, _space: &Space) -> impl Iterator<Item = &'_ usize> + '_
    where
        Space: crate::space::StateActionSpace,
        Self: super::ActionPathFor<Space>,
    {
        self.actions.iter()
    }
}
