use core::num::NonZeroUsize;

pub struct ActionData {
    a: usize,
    next_position: Option<NonZeroUsize>,
    g_sa: Gain,
}

pub enum Gain {
    Predicted(f32),
    Measured(f32),
    Exhausted,
}

impl ActionData {
    pub(crate) fn new_predicted(a: usize, g_sa: f32) -> Self {
        Self {
            a,
            next_position: None,
            g_sa: Gain::Predicted(g_sa),
        }
    }

    pub(crate) fn g_sa(&self) -> Option<f32> {
        match self.g_sa {
            Gain::Predicted(g) => Some(g),
            Gain::Measured(g) => Some(g),
            Gain::Exhausted => None,
        }
    }

    pub(crate) fn action(&self) -> usize {
        // debug_assert!(self.g_sa().is_some());
        self.a
    }

    pub(crate) fn next_position(&self) -> Option<NonZeroUsize> {
        self.next_position
    }

    pub(crate) fn next_position_mut(&mut self) -> &mut Option<NonZeroUsize> {
        debug_assert!(self.g_sa().is_some());
        &mut self.next_position
    }

    pub(crate) fn decay(&mut self, decay: f32, g: f32) {
        match &mut self.g_sa {
            Gain::Predicted(_) => {
                self.g_sa = Gain::Predicted(g);
            },
            Gain::Measured(g_sa) => {
                *g_sa -= decay;
            },
            Gain::Exhausted => unreachable!(),
        }
    }

    pub(crate) fn exhaust(&mut self) {
        // println!("exhausting action {}!", self.a);
        // debug_assert!(self.g_sa().is_some());
        self.g_sa = Gain::Exhausted;
    }

    pub(crate) fn update_g_sa(&mut self, g: f32) {
        match self.g_sa {
            Gain::Predicted(_) | Gain::Measured(_) => {
                self.g_sa = Gain::Measured(g);
            },
            Gain::Exhausted => unreachable!(),
        }
    }
}
