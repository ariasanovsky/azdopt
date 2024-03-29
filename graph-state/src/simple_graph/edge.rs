#[derive(Clone, PartialEq, Eq, Ord, PartialOrd)]
pub struct Edge {
    pub(crate) max: usize,
    pub(crate) min: usize,
}

impl core::fmt::Debug for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { max, min } = self;
        // f.debug_map().entry(&"max", max).entry(&"min", min).finish()
        f.debug_list().entry(max).entry(min).finish()
    }
}

// impl az_discrete_opt::state::cost::CostsOneEach for Edge {}

impl Edge {
    pub const fn new(u: usize, v: usize) -> Self {
        debug_assert!(u != v);
        // todo! stuck behind #![feature(const_trait_impl)]
        // Self { max: u.max(v), min: u.min(v) }
        if u > v {
            Self { max: u, min: v }
        } else {
            Self { max: v, min: u }
        }
    }

    /// # Safety
    /// This function should only be called when `max` > `min`.
    /// It does not perform checks and assumes the caller has already checked that `max` > `min`.
    pub const unsafe fn new_unchecked(max: usize, min: usize) -> Self {
        Self { max, min }
    }

    pub const fn max(&self) -> usize {
        self.max
    }

    pub const fn min(&self) -> usize {
        self.min
    }

    pub const fn vertices(&self) -> (usize, usize) {
        (self.max, self.min)
    }

    pub const fn colex_position(&self) -> usize {
        let (max, min) = self.vertices();
        let last_pos = max * (max + 1) / 2;
        let diff = max - min;
        last_pos - diff
    }

    pub const fn from_colex_position(pos: usize) -> Self {
        let mut v = 1;
        loop {
            let last_position = v * (v + 1) / 2;
            if pos < last_position {
                let diff = last_position - pos;
                return Self::new(v, v - diff);
            }
            v += 1;
        }
    }

    pub fn edges<const N: usize>() -> impl Iterator<Item = Self> {
        (0..N).flat_map(|v| (0..v).map(move |u| unsafe { Self::new_unchecked(v, u) }))
    }
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn colex_position_of_edges_on_ten_vertices_are_correct() {
        let edges = (0..10).flat_map(|v| (0..v).map(move |u| Edge::new(v, u)));
        edges.enumerate().for_each(|(i, e)| {
            assert_eq!(e.colex_position(), i);
            assert_eq!(Edge::from_colex_position(i), e);
        });
    }
}
