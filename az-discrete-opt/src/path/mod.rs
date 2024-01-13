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
    fn actions_taken(&self) -> impl Iterator<Item = usize> + '_;
}

/// # Safety
/// `Self` must be a valid `ActionPath` for `Space`
pub unsafe trait ActionPathFor<Space>: ActionPath {}
