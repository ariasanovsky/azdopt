use core::num::NonZeroUsize;
use std::ops::SubAssign;

pub struct ActionData {
    a: usize,
    next_position: Option<NonZeroUsize>,
    g_sa: Option<f32>,
}

impl ActionData {
    pub(crate) fn new(a: usize, g_sa: f32) -> Self {
        Self {
            a,
            next_position: None,
            g_sa: Some(g_sa),
        }
    }

    pub(crate) fn g_sa(&self) -> Option<f32> {
        self.g_sa
    }

    pub(crate) fn action(&self) -> usize {
        self.a
    }

    pub(crate) fn next_position(&self) -> Option<NonZeroUsize> {
        self.next_position
    }

    pub(crate) fn next_position_mut(&mut self) -> &mut Option<NonZeroUsize> {
        &mut self.next_position
    }

    pub(crate) fn decay(&mut self) {
        let g_sa = self.g_sa.as_mut().unwrap();
        g_sa.sub_assign(0.05);
    }

    pub(crate) fn exhaust(&mut self) {
        debug_assert!(self.g_sa.is_some());
        // println!("exhausting action {}!", self.a);
        self.g_sa = None;
    }

    pub(crate) fn update_g_sa(&mut self, g: f32) {
        let g_sa = self.g_sa.as_mut().unwrap();
        *g_sa = g_sa.max(g);
    }
}
