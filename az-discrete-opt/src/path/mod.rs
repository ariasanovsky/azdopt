use crate::space::StateActionSpace;

pub mod sequence;
pub mod ord_set;
pub mod multiset;
pub mod set;

pub trait ActionPath {
    fn new() -> Self;
    fn len(&self) -> usize;
    unsafe fn push_unchecked(&mut self, action: usize);
}

pub unsafe trait ActionPathFor<Space: StateActionSpace>: ActionPath {
    fn push(&mut self, action: &Space::Action) {
        let index = Space::index(action);
        unsafe { self.push_unchecked(index) }
    }
}
