#[derive(Clone, PartialEq, Eq)]
pub struct Edge {
    pub max: usize,
    pub min: usize,
}

impl Edge {
    pub fn new(u: usize, v: usize) -> Self {
        assert_ne!(u, v);
        Self { max: u.max(v), min: u.min(v) }
    }

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
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn colex_position_of_edges_on_ten_vertices_are_correct() {
        let edges = (0..10).flat_map(|v| (0..v).map(move |u| Edge::new(v, u)));
        edges.enumerate().for_each(|(i, e)| {
            assert_eq!(e.colex_position(), i);
        });
    }
}