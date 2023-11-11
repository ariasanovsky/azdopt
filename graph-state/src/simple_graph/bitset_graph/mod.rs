use az_discrete_opt::state::StateNode;
use core::mem::MaybeUninit;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::bitset::primitive::B32;
use crate::bitset::bitset::Bitset;

use super::edge::Edge;

mod display;
mod graph6;
pub(crate) mod state;
mod try_from;

#[derive(Clone, Debug)]
pub struct BitsetGraph<const N: usize, B = B32> {
    pub(crate) neighborhoods: [B; N],
}

trait AllVertices {
    type B;
    const ALL_VERTICES: Self::B;
}

impl<const N: usize> AllVertices for BitsetGraph<N> {
    type B = B32;
    const ALL_VERTICES: B32 = B32::from_bits((1 << N) - 1);
}

impl<const N: usize> BitsetGraph<N> {
    // const ALL_VERTICES: B32 = B32::new((1 << N) - 1);

    pub fn par_generate_batch<const BATCH: usize>(time: usize, p: f64) -> [StateNode<Self>; BATCH] {
        // todo! size asserts, move par_generate_batch to `StateNode` impl block
        let mut states: [MaybeUninit<StateNode<Self>>; BATCH] = MaybeUninit::uninit_array();
        states.par_iter_mut().for_each(|s| {
            let mut rng = rand::thread_rng();
            s.write(StateNode::new(Self::generate(p, &mut rng), time));
        });
        unsafe { MaybeUninit::array_assume_init(states) }
    }

    pub fn generate(p: f64, rng: &mut impl rand::Rng) -> Self {
        let mut neighborhoods = core::array::from_fn(|_| B32::empty());
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

    pub fn is_connected(&self) -> bool {
        if N == 0 {
            return true;
        }
        let Self { neighborhoods } = self;
        let mut visited_vertices = B32::empty();
        unsafe { visited_vertices.add_unchecked(0) };
        let mut seen_vertices = neighborhoods[0].clone();
        while !seen_vertices.is_empty() {
            seen_vertices = seen_vertices
                .iter()
                .map(|v| {
                    unsafe { visited_vertices.add_unchecked(v as _) };
                    &neighborhoods[v]
                })
                .fold(B32::empty(), |mut acc, n| {
                    acc.union_assign(n);
                    acc
                });
            seen_vertices.minus_assign(&visited_vertices);
        }
        visited_vertices == Self::ALL_VERTICES
    }

    pub fn to_connected(self) -> Option<super::connected_bitset_graph::ConnectedBitsetGraph<N>> {
        if self.is_connected() {
            Some(super::connected_bitset_graph::ConnectedBitsetGraph {
                neighborhoods: self.neighborhoods,
            })
        } else {
            None
        }
    }

    pub fn edges(&self) -> impl Iterator<Item = Edge> + '_ {
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

    pub fn edge_bools(&self) -> impl Iterator<Item = bool> + '_ {
        let Self { neighborhoods } = self;
        neighborhoods
            .iter()
            .enumerate()
            .flat_map(move |(v, n)| (0..v).map(move |u| n.contains(u as _).unwrap()))
    }
}
