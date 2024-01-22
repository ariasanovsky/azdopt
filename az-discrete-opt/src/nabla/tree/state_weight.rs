use core::{num::NonZeroU32, ops::Range};

#[derive(Debug)]
pub struct StateWeight {
    pub(crate) c: f32,
    pub(crate) c_t_star: f32,
    pub(crate) n_t: NumLeafDescendants,
    pub(crate) actions: Range<u32>,
}

#[derive(Debug, Clone, Copy)]
pub(crate) struct NumLeafDescendants {
    num_leaves: u32,
}

#[derive(Debug, Clone, Copy, Eq, PartialEq, PartialOrd, Ord)]
pub(crate) struct ActiveNumLeafDescendents {
    num_leaves: NonZeroU32,
}

impl NumLeafDescendants {
    const ACTIVE_BIT: u32 = 1;

    pub(crate) fn mark_exhausted(&mut self) {
        debug_assert!(self.is_active());
        self.num_leaves ^= Self::ACTIVE_BIT;
    }

    pub(crate) fn is_active(&self) -> bool {
        self.num_leaves & Self::ACTIVE_BIT == 1
    }

    pub(crate) fn try_active(&self) -> Option<ActiveNumLeafDescendents> {
        match self.is_active() {
            true => Some(ActiveNumLeafDescendents {
                num_leaves: unsafe { NonZeroU32::new_unchecked(self.num_leaves) },
            }),
            false => None,
        }
    }

    pub(crate) fn increment(&mut self) {
        self.num_leaves += 2;
    }

    pub(crate) fn value(&self) -> u32 {
        self.num_leaves / 2
    }

    pub(crate) fn join(&self, other: &Self) -> Self {
        let active_bit = (self.num_leaves | other.num_leaves) & Self::ACTIVE_BIT;
        let max = self.num_leaves.max(other.num_leaves);
        Self {
            num_leaves: max | active_bit,
        }
    }
}

impl ActiveNumLeafDescendents {
    pub(crate) fn value(&self) -> u32 {
        self.num_leaves.get() / 2
    }
}

impl StateWeight {
    pub(crate) fn new(c: f32) -> Self {
        Self {
            c,
            c_t_star: c,
            n_t: NumLeafDescendants { num_leaves: 1 },
            actions: 0..0,
        }
    }

    pub(crate) fn mark_exhausted(&mut self) {
        // debug_assert!(self.n_t.is_some());
        self.n_t.mark_exhausted();
    }

    pub fn c(&self) -> f32 {
        self.c
    }

    pub fn c_star(&self) -> f32 {
        self.c_t_star
    }
}
