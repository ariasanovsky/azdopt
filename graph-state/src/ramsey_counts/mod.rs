use az_discrete_opt::{log::ShortForm, state::cost::Cost};
use itertools::Itertools;

use crate::{simple_graph::bitset_graph::ColoredCompleteBitsetGraph, bitset::Bitset};

pub mod space;

#[derive(Clone, Debug)]
pub struct RamseyCounts<const N: usize, const E: usize, const C: usize, B> {
    graph: ColoredCompleteBitsetGraph<N, C, B>,
    counts: [[i32; E]; C],
    total_counts: TotalCounts<C>,
}

impl<const N: usize, const E: usize, const C: usize, B> RamseyCounts<N, E, C, B> {
    pub fn new(graph: ColoredCompleteBitsetGraph<N, C, B>, sizes: &[usize; C]) -> Self
    where
        B: Bitset + Clone,
        B::Bits: Clone,
    {
        let mut counts = [[0; E]; C];
        let mut total_counts = [0; C];
        enum Intersection<B> {
            Adjacent(B),
            NotAdjacent(B),
        }
        for (((&size, counts), g_i), total) in sizes.iter().zip(counts.iter_mut()).zip(graph.graphs().iter()).zip(total_counts.iter_mut()) {
            let common_neighbors = g_i.neighborhoods.iter().enumerate().flat_map(|(v, n_v)| {
                (0..v).map(move |u| {
                    let intersection = n_v.intersection(&g_i.neighborhoods[u]);
                    if unsafe { n_v.contains_unchecked(u as _) } {
                        Intersection::Adjacent(intersection)
                    } else {
                        Intersection::NotAdjacent(intersection)
                    }
                })
            });
            for (common_neighbors, count) in common_neighbors.zip_eq(counts.iter_mut()) {
                *count = match common_neighbors {
                    Intersection::Adjacent(common_neighbors) => {
                        let count = g_i.count_cliques_inside(common_neighbors, size - 2);
                        *total += count;
                        count
                    },
                    Intersection::NotAdjacent(common_neighbors) => {
                        g_i.count_cliques_inside(common_neighbors, size - 2)
                    },
                }
            }
        }
        Self { graph, counts, total_counts: TotalCounts(total_counts) }
    }

    pub fn graph(&self) -> &ColoredCompleteBitsetGraph<N, C, B> {
        &self.graph
    }

    pub fn clique_counts(&self) -> &TotalCounts<C> {
        &self.total_counts
    }
}

pub struct AssignColor {

}

impl AssignColor {
    
}

pub struct CountChange {

}

#[derive(Clone, Debug)]
pub struct TotalCounts<const C: usize>(pub [i32; C]);

impl<const N: usize, const E: usize, const C: usize, B> ShortForm for RamseyCounts<N, E, C, B>
where
    B: Bitset,
    B::Bits: Clone,
{
    fn short_form(&self) -> String {
        let graphs = self.graph.graphs();
        let counts = self.clique_counts().0;
        graphs.iter().zip(counts.iter()).map(|(g, c)| format!("{}: {}", g, c)).join(", ")
    }
}