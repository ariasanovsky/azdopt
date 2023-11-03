use core::mem::MaybeUninit;

use az_discrete_opt::state::StateNode;
use faer::{Mat, Faer};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

use crate::bitset::B32;

use super::{bitset_graph::BitsetGraph, edge::Edge};

mod display;
mod graph6;
mod state;

#[derive(Clone, Debug)]
pub struct ConnectedBitsetGraph<const N: usize, B = B32> {
    pub(crate) neighborhoods: [B; N],
}

pub enum ActionKind {
    Add,
    Delete,
}

impl<const N: usize> ConnectedBitsetGraph<N> {
    pub fn is_cut_edge(&self, e: &Edge) -> bool {
        let Self { neighborhoods } = self;
        let (v, u) = e.vertices();
        let mut new_vertices = neighborhoods[v].clone();
        // if `uv` is not an edge, then `new_vertices` is nonempty and contains `u`
        // so we don't need to check if `u` is in `new_vertices`
        new_vertices.add_or_remove_unchecked(u);
        let mut explored_vertices = B32::empty();
        explored_vertices.insert_unchecked(v);
        while !new_vertices.is_empty() {
            if new_vertices.contains(u) {
                return false;
            }
            explored_vertices.union_assign(&new_vertices);
            let recently_seen_vertices = new_vertices;
            new_vertices = B32::empty();
            recently_seen_vertices.iter().for_each(|v| {
                new_vertices.union_assign(&neighborhoods[v]);
            });
            new_vertices.minus_assign(&explored_vertices);
        }
        true
    }


    pub fn par_generate_batch<const BATCH: usize>(time: usize, p: f64) -> [StateNode<Self>; BATCH] {
        // todo! size asserts, move to `StateNode`
        let mut states: [MaybeUninit<StateNode<Self>>; BATCH] = MaybeUninit::uninit_array();
        states.par_iter_mut().for_each(|s| {
            let mut rng = rand::thread_rng();
            s.write(StateNode::new(Self::generate(p, &mut rng), time));
        });
        let states = unsafe { MaybeUninit::array_assume_init(states) };
        states
    }

    pub fn generate(p: f64, rng: &mut impl rand::Rng) -> Self {
        loop {
            let graph = BitsetGraph::<N>::generate(p, rng);
            if graph.is_connected() {
                return Self { neighborhoods: graph.neighborhoods };
            }
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

    pub fn num_edges(&self) -> u32 {
        let Self {
            neighborhoods,
        } = self;
        neighborhoods.iter().map(|n| n.cardinality()).sum::<u32>() / 2
    }

    pub fn edge_bools(&self) -> impl Iterator<Item = bool> + '_ {
        let Self { neighborhoods } = self;
        neighborhoods.iter().enumerate().flat_map(move |(v, n)| {
            (0..v).map(move |u| n.contains(u))
        })
    }

    pub fn action_kinds(&self) -> impl Iterator<Item = Option<ActionKind>> + '_ {
        let Self { neighborhoods } = self;
        neighborhoods.iter().enumerate().flat_map(move |(v, n)| {
            (0..v).map(move |u| {
                if n.contains(u) {
                    let e = unsafe { Edge::new_unchecked(v, u) };
                    if self.is_cut_edge(&e) {
                        None
                    } else {
                        Some(ActionKind::Delete)
                    }
                } else {
                    Some(ActionKind::Add)
                }
            })
        })
    }

    pub fn ah_cost(&self) -> f32 {
        let mut distances: [[usize; N]; N] = [[0; N]; N];
        let mut diameter = 0;
        let min_transmission = (0..N).map(|u| {
            let mut explored = B32::empty();
            explored.insert_unchecked(u);
            let mut newly_seen_vertices = self.neighborhoods[u].clone();
            let mut transmission = 0;
            for d in 1.. {
                if newly_seen_vertices.is_empty() {
                    diameter = diameter.max(d - 1);
                    break;
                }
                transmission += newly_seen_vertices.cardinality() * d;
                explored.union_assign(&newly_seen_vertices);
                let recently_seen_vertices = newly_seen_vertices;
                newly_seen_vertices = B32::empty();
                recently_seen_vertices.iter().for_each(|v| {
                    distances[u][v] = d as usize;
                    newly_seen_vertices.union_assign(&self.neighborhoods[v]);
                });
                newly_seen_vertices.minus_assign(&explored);
            }
            transmission
        }).min().unwrap();
        let proximity = min_transmission as f64 / (N - 1) as f64;
        let k = (2 * diameter / 3).checked_sub(1).map(|k| k as usize).unwrap_or(N - 1);
        let a: Mat<f64> = Mat::from_fn(N, N, |i, j| distances[i][j] as f64);
        let mut eigs = a.selfadjoint_eigenvalues(faer::Side::Lower);
        eigs.sort_floats();
        eigs.reverse();
        (proximity + eigs[k]) as f32
    }
}
