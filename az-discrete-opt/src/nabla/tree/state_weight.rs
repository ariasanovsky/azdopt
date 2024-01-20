use core::{ops::Range, num::NonZeroU32};

#[derive(Debug)]
pub struct StateWeight {
    pub(crate) c: f32,
    pub(crate) c_t_star: f32,
    pub(crate) n_t: Option<NonZeroU32>,
    pub(crate) actions: Range<usize>,
}

impl StateWeight {
    pub(crate) fn new(c: f32) -> Self {
        Self {
            c,
            c_t_star: c,
            n_t: Some(NonZeroU32::MIN),
            actions: 0..0,
        }
    }

    pub(crate) fn assert_exhausted(&mut self) {
        self.n_t = None;
    }
}
