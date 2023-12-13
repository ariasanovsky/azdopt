use crate::space::StateActionSpace;

pub mod multiset;
pub mod ord_set;
pub mod sequence;
pub mod set;

pub trait ActionPath {
    fn new() -> Self;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn clear(&mut self);
    /// # Safety
    /// `action` must be a valid index for `Space`
    unsafe fn push_unchecked(&mut self, action: usize);
    fn push<Space>(&mut self, space: &Space, action: &Space::Action)
    where
        Space: StateActionSpace,
        Self: ActionPathFor<Space>,
    {
        let index = space.index(action);
        unsafe { self.push_unchecked(index) }
    }
    fn actions_taken<Space>(&self, space: &Space) -> impl Iterator<Item = &'_ usize> + '_
    where
        Space: StateActionSpace,
        Self: ActionPathFor<Space>;
}

/// # Safety
/// `Self` must be a valid `ActionPath` for `Space`
pub unsafe trait ActionPathFor<Space: StateActionSpace>: ActionPath {}
