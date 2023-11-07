use core::mem::MaybeUninit;
use std::collections::VecDeque;

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

    pub fn adjacency_matrix(&self) -> Mat<f64> {
        let mut a = Mat::zeros(N, N);
        for (v, n) in self.neighborhoods.iter().enumerate() {
            for u in n.iter() {
                a[(v, u)] = 1.0;
            }
        }
        a
    }

    pub fn matching_number(&self) -> usize {
        self.maximum_matching().len()
    }

    pub fn maximum_matching(&self) -> Vec<Edge> {
        struct MatchingSearch {
            edges: Vec<Edge>,
            unvisited_vertices: B32,
        }
        if N == 0 {
            return Vec::with_capacity(0);
        }
        let first_matching = MatchingSearch {
            edges: Vec::new(),
            unvisited_vertices: B32::range_to_unchecked(N),
        };
        let mut matching_queue = VecDeque::new();
        matching_queue.push_back(first_matching);
        let mut matching_number = 0;
        let mut max_matching = Vec::with_capacity(0);
        while let Some(matching) = matching_queue.pop_back() {
            let MatchingSearch { edges, mut unvisited_vertices } = matching;
            for edge in edges.iter() {
                assert!(
                    !unvisited_vertices.contains(edge.min()),
                    "I think this is impossible, let's see if it ever fails"
                );
                assert!(
                    !unvisited_vertices.contains(edge.max()),
                    "I think this is impossible, let's see if it ever fails"
                );
            }
            let max_future_increase = unvisited_vertices.cardinality() / 2;
            if edges.len() + max_future_increase as usize <= matching_number {
                continue;
            }
            assert!(
                !unvisited_vertices.is_empty(),
                "I think this is impossible, let's see if it ever fails"
            );
            let next_v = unvisited_vertices.max_unchecked();
            assert!(
                unvisited_vertices.contains(next_v),
                "I think this is impossible, let's see if it ever fails"
            );
            unvisited_vertices.add_or_remove_unchecked(next_v);
            let max_future_increase = unvisited_vertices.cardinality() / 2;
            // we push this matching first so that our search is a DFS
            // this will let us get to large matching first
            // as a consequence, more matchings will be skipped
            if edges.len() + max_future_increase as usize > matching_number {
                let new_matching = MatchingSearch {
                    edges: edges.clone(),
                    unvisited_vertices: unvisited_vertices.clone(),
                };
                matching_queue.push_back(new_matching);
            }
            let mut unvisited_v_neighbors = unvisited_vertices.clone();
            unvisited_v_neighbors.intersection_assign(&self.neighborhoods[next_v]);
            for next_u in unvisited_v_neighbors.iter() {
                let mut new_edges = edges.clone();
                new_edges.push(unsafe { Edge::new_unchecked(next_v, next_u) });
                let mut new_unvisited_vertices = unvisited_vertices.clone();
                assert!(
                    new_unvisited_vertices.contains(next_u),
                    "I think this is impossible, let's see if it ever fails"
                );
                new_unvisited_vertices.add_or_remove_unchecked(next_u);
                if new_edges.len() > matching_number {
                    matching_number = new_edges.len();
                    max_matching = new_edges.clone();
                }
                // new_unvisited_vertices.minus_assign(&self.neighborhoods[next_u]);
                let max_future_increase = new_unvisited_vertices.cardinality() / 2;
                if new_edges.len() + max_future_increase as usize > matching_number {
                    let new_matching = MatchingSearch {
                        edges: new_edges,
                        unvisited_vertices: new_unvisited_vertices,
                    };
                    matching_queue.push_back(new_matching);
                }
            }
        }
        dbg!(&max_matching);
        let mut vxs = max_matching.iter().map(|e: &Edge| core::iter::once(e.max()).chain(core::iter::once(e.min()))).flatten().collect::<Vec<_>>();
        vxs.sort();
        dbg!(&vxs);
        max_matching
    }

    pub fn conjecture_2_1_cost(&self) -> (Vec<Edge>, f64) {
        let a = self.adjacency_matrix();
        // println!("{a:?}");
        // let eigs = a.selfadjoint_eigenvalues(faer::Side::Lower);
        // let lambda_1 = eigs.into_iter().max_by(|a, b| a.partial_cmp(&b).unwrap()).unwrap();
        let eigs: Vec<faer::complex_native::c64> = a.eigenvalues();
        let lambda_1 = eigs.into_iter().max_by(|a, b| a.re.partial_cmp(&b.re).unwrap()).unwrap().re;
        let matching = self.maximum_matching();
        (matching, lambda_1)
    }
}

#[cfg(test)]
mod tests {

    use super::*;

    #[test]
    fn complete_graph_on_four_vertices_has_matching_number_two() {
        let graph: BitsetGraph<4> = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)].as_ref().try_into().unwrap();
        let graph = graph.to_connected().unwrap();
        assert_eq!(graph.matching_number(), 2);
    }

    #[test]
    fn cycle_graph_on_five_vertices_has_matching_number_two() {
        let graph: BitsetGraph<5> = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)].as_ref().try_into().unwrap();
        let graph = graph.to_connected().unwrap();
        assert_eq!(graph.matching_number(), 2);
    }

    #[test]
    fn this_one_tree_on_twenty_vertices_has_matching_number_nine() {
        let graph: BitsetGraph<20> = [(0, 11), (0, 16), (0, 19), (1, 15), (1, 17), (2, 13), (3, 14), (4, 13), (4, 14), (5, 9), (5, 10), (5, 18), (6, 15), (7, 17), (7, 19), (8, 10), (9, 12), (10, 13), (16, 18)].as_ref().try_into().unwrap();
        let graph = graph.to_connected().unwrap();
        assert_eq!(graph.matching_number(), 9);
    }
}
