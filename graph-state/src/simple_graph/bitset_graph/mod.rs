use core::mem::MaybeUninit;
use az_discrete_opt::state::StateNode;
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::bitset::B32;

use super::edge::Edge;

pub(crate) mod state;
mod display;
mod graph6;
mod try_from;

#[derive(Clone, Debug)]
pub struct BitsetGraph<const N: usize, B = B32> {
    pub(crate) neighborhoods: [B; N],
}

impl<const N: usize> BitsetGraph<N> {
    const ALL_VERTICES: B32 = B32::new((1 << N) - 1);

    pub fn par_generate_batch<const BATCH: usize>(time: usize, p: f64) -> [StateNode<Self>; BATCH] {
        // todo! size asserts, move par_generate_batch to `StateNode` impl block
        let mut states: [MaybeUninit<StateNode<Self>>; BATCH] = MaybeUninit::uninit_array();
        states.par_iter_mut().for_each(|s| {
            let mut rng = rand::thread_rng();
            s.write(StateNode::new(Self::generate(p, &mut rng), time));
        });
        let states = unsafe { MaybeUninit::array_assume_init(states) };
        states
    }

    pub fn generate(p: f64, rng: &mut impl rand::Rng) -> Self {
        let mut neighborhoods = core::array::from_fn(|_| B32::empty());
        for v in 0..N {
            for u in 0..v {
                if rng.gen_bool(p) {
                    neighborhoods[v].add_or_remove_unchecked(u);
                    neighborhoods[u].add_or_remove_unchecked(v);
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
        visited_vertices.insert_unchecked(0);
        let mut seen_vertices = neighborhoods[0].clone();
        while !seen_vertices.is_empty() {
            seen_vertices = seen_vertices.iter().map(|v| {
                visited_vertices.insert_unchecked(v);
                &neighborhoods[v]
            }).fold(B32::empty(), |mut acc, n| {
                acc.union_assign(n);
                acc
            });
            seen_vertices.minus_assign(&visited_vertices);
        }
        visited_vertices == Self::ALL_VERTICES
    }

    pub fn to_connected(self) -> Option<super::connected_bitset_graph::ConnectedBitsetGraph<N>> {
        if self.is_connected() {
            Some(super::connected_bitset_graph::ConnectedBitsetGraph { neighborhoods: self.neighborhoods })
        } else {
            None
        }
    }

    pub fn edges(&self) -> impl Iterator<Item = Edge> + '_ {
        let Self { neighborhoods } = self;
        neighborhoods.iter().enumerate().flat_map(move |(v, n)| {
            (0..v).filter_map(move |u| {
                let e = unsafe { Edge::new_unchecked(v, u) };
                if n.contains(u) {
                    Some(e)
                } else {
                    None
                }
            })
        })
    }

    pub fn edge_bools(&self) -> impl Iterator<Item = bool> + '_ {
        let Self { neighborhoods } = self;
        neighborhoods.iter().enumerate().flat_map(move |(v, n)| {
            (0..v).map(move |u| n.contains(u))
        })
    }
}
