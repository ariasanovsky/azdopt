use std::collections::BTreeSet;

use rand::seq::SliceRandom;

use crate::bitset::Bitset;

use super::RamseyCounts;

#[derive(Clone, Debug)]
pub struct RamseyCountsNoRecolor<const N: usize, const E: usize, const C: usize, B> {
    pub counts: RamseyCounts<N, E, C, B>,
    pub modifiable_edges: BTreeSet<usize>,
}

impl<const N: usize, const E: usize, const C: usize, B> RamseyCountsNoRecolor<N, E, C, B> {
    const EDGE_POSITIONS: [usize; E] = {
        let mut edges = [0; E];
        let mut i = 0;
        while i < E {
            edges[i] = i;
            i += 1;
        }
        edges
    };

    pub fn choose_edges(num_edges: usize, rng: &mut impl rand::Rng) -> impl Iterator<Item = usize> {
        Self::EDGE_POSITIONS.choose_multiple(rng, num_edges).copied()
    }

    pub fn randomize_modifiable_edges(&mut self, num_edges: usize, rng: &mut impl rand::Rng) {
        let edges = Self::EDGE_POSITIONS.choose_multiple(rng, num_edges);
        self.modifiable_edges = edges.copied().collect();
    }

    pub fn generate(
        rng: &mut impl rand::Rng,
        w: &impl rand::distributions::Distribution<usize>,
        sizes: &[usize; C],
        num_edges: usize,
    ) -> Self
    where
        B: Bitset + Clone,
        B::Bits: Clone,
    {
        let counts = RamseyCounts::generate(rng, w, sizes);
        let modifiable_edges = Self::choose_edges(num_edges, rng).collect();
        Self { counts, modifiable_edges }
    }

}