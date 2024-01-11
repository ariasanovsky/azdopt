use std::collections::BTreeSet;

use rand::seq::SliceRandom;

use crate::bitset::Bitset;

use super::RamseyCounts;

#[derive(Clone, Debug)]
pub struct RamseyCountsNoRecolor<const N: usize, const E: usize, const C: usize, B> {
    pub(crate) state: RamseyCounts<N, E, C, B>,
    pub(crate) permitted_edges: BTreeSet<usize>,
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

    pub fn randomize_permitted_edges(
        &mut self,
        num_permitted_edges: usize,
        rng: &mut impl rand::Rng,
    ) {
        let prohibited_edges = Self::EDGE_POSITIONS
            .choose_multiple(rng, num_permitted_edges)
            .copied();
        self.permitted_edges = prohibited_edges.collect();
    }

    pub fn generate(
        rng: &mut impl rand::Rng,
        counts: RamseyCounts<N, E, C, B>,
        num_permitted_edges: usize,
    ) -> Self
    where
        B: Bitset + Clone,
        B::Bits: Clone,
    {
        let permitted_edges = Self::EDGE_POSITIONS
            .choose_multiple(rng, num_permitted_edges)
            .copied()
            .collect();
        Self {
            state: counts,
            permitted_edges,
        }
    }

    pub fn num_permitted_edges(&self) -> usize {
        self.permitted_edges.len()
    }
}
