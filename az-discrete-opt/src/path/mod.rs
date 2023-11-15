use crate::space::StateActionSpace;

mod sequence;
mod ord_set;
mod multiset;
mod set;

pub trait ActionPath {
    fn len(&self) -> usize;
    unsafe fn push_unchecked(&mut self, action: usize);
}

pub unsafe trait ActionPathFor<Space: StateActionSpace>: ActionPath {
    fn push(&mut self, action: &Space::Action) {
        let index = Space::index(action);
        unsafe { self.push_unchecked(index) }
    }
}
