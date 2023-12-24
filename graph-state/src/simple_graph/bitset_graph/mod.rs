use crate::bitset::Bitset;

use super::edge::Edge;

mod display;
mod graph6;
pub mod space;
mod try_from;

#[derive(Clone, Debug)]
pub struct BitsetGraph<const N: usize, B> {
    pub(crate) neighborhoods: [B; N],
}

#[derive(Clone, Debug)]
pub struct ColoredCompleteBitsetGraph<const N: usize, const C: usize, B> {
    pub(crate) graphs: [BitsetGraph<N, B>; C],
}

impl<const N: usize, const C: usize, B> ColoredCompleteBitsetGraph<N, C, B> {
    pub fn generate(
        w: &impl rand::distributions::Distribution<usize>,
        rng: &mut impl rand::Rng,
    ) -> ColoredCompleteBitsetGraph<N, C, B>
    where
        B: Bitset,
    {
        debug_assert!(N < 32);
        let mut graphs: [BitsetGraph<N, B>; C] = core::array::from_fn(|_| BitsetGraph::empty());
        let edges =
            (0..N)
            .flat_map(|v| (0..v).map(move |u| unsafe { Edge::new_unchecked(v, u) }));
        for e in edges {
            let c = w.sample(rng);
            let g = &mut graphs[c];
            unsafe { g.add_or_remove_edge_unchecked(e) };
        }
        Self { graphs }
    }

    pub fn graphs(&self) -> &[BitsetGraph<N, B>; C] {
        &self.graphs
    }

    pub fn color(&self, u: usize, v: usize) -> usize
    where
        B: Bitset,
    {
        let c = self.graphs()
            .iter()
            .map(|g| &g.neighborhoods[u])
            .position(|n| n.contains(v as _).unwrap())
            .unwrap();
        c
    }
}

impl<const N: usize, B> BitsetGraph<N, B> {
    pub unsafe fn add_or_remove_edge_unchecked(&mut self, e: Edge)
    where
        B: Bitset,
    {
        let (v, u) = e.vertices();
        self.neighborhoods[v].add_or_remove_unchecked(u as u32);
        self.neighborhoods[u].add_or_remove_unchecked(v as u32);
    }

    pub fn empty() -> Self
    where
        B: Bitset,
    {
        Self {
            neighborhoods: core::array::from_fn(|_| B::empty()),
        }
    }
    
    pub fn generate(p: f64, rng: &mut impl rand::Rng) -> Self
    where
        B: Bitset,
    {
        let mut neighborhoods = core::array::from_fn(|_| B::empty());
        for v in 0..N {
            for u in 0..v {
                if rng.gen_bool(p) {
                    unsafe { neighborhoods[v].add_or_remove_unchecked(u as u32) };
                    unsafe { neighborhoods[u].add_or_remove_unchecked(v as u32) };
                }
            }
        }
        Self { neighborhoods }
    }

    pub fn is_connected(&self) -> bool
    where
        B: Bitset + Clone + PartialEq,
        B::Bits: Clone,
    {
        if N == 0 {
            return true;
        }
        let Self { neighborhoods } = self;
        let mut visited_vertices = B::empty();
        unsafe { visited_vertices.add_unchecked(0) };
        let mut seen_vertices = neighborhoods[0].clone();
        while !seen_vertices.is_empty() {
            seen_vertices = seen_vertices
                .iter()
                .map(|v| {
                    unsafe { visited_vertices.add_unchecked(v as _) };
                    &neighborhoods[v]
                })
                .fold(B::empty(), |mut acc, n| {
                    acc.union_assign(n);
                    acc
                });
            seen_vertices.minus_assign(&visited_vertices);
        }
        visited_vertices == unsafe { B::range_to_unchecked(N as _) }
    }

    pub fn to_connected(self) -> Option<super::connected_bitset_graph::ConnectedBitsetGraph<N, B>>
    where
        B: Bitset + Clone + PartialEq,
        B::Bits: Clone,
    {
        if self.is_connected() {
            Some(super::connected_bitset_graph::ConnectedBitsetGraph {
                neighborhoods: self.neighborhoods,
            })
        } else {
            None
        }
    }

    pub fn edges(&self) -> impl Iterator<Item = Edge> + '_
    where
        B: Bitset,
    {
        let Self { neighborhoods } = self;
        neighborhoods.iter().enumerate().flat_map(move |(v, n)| {
            (0..v).filter_map(move |u| {
                let e = unsafe { Edge::new_unchecked(v, u) };
                if n.contains(u as _).unwrap() {
                    Some(e)
                } else {
                    None
                }
            })
        })
    }

    pub fn edge_bools(&self) -> impl Iterator<Item = bool> + '_
    where
        B: Bitset,
    {
        let Self { neighborhoods } = self;
        neighborhoods
            .iter()
            .enumerate()
            .flat_map(move |(v, n)| (0..v).map(move |u| n.contains(u as _).unwrap()))
    }

    pub fn count_cliques_inside(&self, common_neighbors: B, size: usize) -> i32
    where
        B: Bitset + Clone,
        B::Bits: Clone,
    {
        match size {
            0 => 1,
            1 => common_neighbors.cardinality() as i32,
            _ => common_neighbors.iter().map(|u| {
                let common_neighbors = common_neighbors.intersection(&self.neighborhoods[u]);
                self.count_cliques_inside(common_neighbors, size - 1)
            }).sum(),
        }
    }
}
