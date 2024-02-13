use core::ops::Range;

#[derive(Debug)]
pub struct StateWeight {
    pub(crate) c: f32,
    pub(crate) c_t_star: f32,
    pub(crate) n_t: u32,
    pub(crate) exhausted_children: u32,
    pub(crate) actions: Range<u32>,
}

impl StateWeight {
    pub(crate) fn new(c: f32) -> Self {
        Self {
            c,
            c_t_star: c,
            n_t: 0,
            exhausted_children: 0,
            actions: 0..0,
        }
    }

    pub fn c(&self) -> f32 {
        self.c
    }

    pub fn c_star(&self) -> f32 {
        self.c_t_star
    }

    pub(crate) fn is_active(&self) -> bool {
        self.actions.start + self.exhausted_children < self.actions.end
    }

    pub(crate) fn n_t(&self) -> u32 {
        self.n_t
    }
}
