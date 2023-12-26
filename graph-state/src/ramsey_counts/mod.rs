use az_discrete_opt::log::ShortForm;
use itertools::Itertools;

use crate::{simple_graph::{bitset_graph::ColoredCompleteBitsetGraph, edge::Edge}, bitset::Bitset};

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
        for (size, total) in sizes.iter().zip(total_counts.iter_mut()) {
            *total /= (*size * (*size - 1) / 2) as i32;
        }
        Self { graph, counts, total_counts: TotalCounts(total_counts) }
    }

    pub fn graph(&self) -> &ColoredCompleteBitsetGraph<N, C, B> {
        &self.graph
    }

    pub fn clique_counts(&self) -> &TotalCounts<C> {
        &self.total_counts
    }

    pub fn reassign_color(&mut self, edge: Edge, new_color: usize, sizes: &[usize; C])
    where
        B: Bitset + Clone,
        B::Bits: Clone,
    {
        let (v, u) = edge.vertices();
        let old_color = self.graph.color(u, v);
        let old_size = sizes[old_color];
        let new_size = sizes[new_color];
        unsafe { self.graph.graphs[old_color].add_or_remove_edge_unchecked(edge.clone()) };
        self.reassign_color_count_adjustment::<true>(u, v, old_color, old_size);
        self.reassign_color_count_adjustment::<false>(u, v, new_color, new_size);
        unsafe { self.graph.graphs[new_color].add_or_remove_edge_unchecked(edge.clone()) };
        let edge_pos = edge.colex_position();
        let old_count = self.counts[old_color][edge_pos];
        let new_count = self.counts[new_color][edge_pos];
        self.total_counts.0[old_color] -= old_count;
        self.total_counts.0[new_color] += new_count;
        debug_assert!(self.total_counts.0.iter().all(|&c| c >= 0));
    }

    fn reassign_color_count_adjustment<const SUBTRACT: bool>(
        &mut self,
        u: usize,
        v: usize,
        color: usize,
        size: usize,
    )
    where
        B: Bitset + Clone,
        B::Bits: Clone,
    {
        if N <= 2 {
            return
        }
        let counts = &mut self.counts[color];
        let graph = &self.graph.graphs()[color];
        let n_u = graph.neighborhoods[u].clone();
        let n_v = graph.neighborhoods[v].clone();
        let n_uv = n_u.intersection(&n_v);
        // edge `{v, w}` (requires `w` in `n_u`)
        for w in n_u.iter() {
            let n_uvw = n_uv.intersection(&graph.neighborhoods[w]);
            let count_change = graph.count_cliques_inside(n_uvw, size - 3);
            if count_change != 0 {
                let vw_pos = Edge::new(v, w).colex_position();
                if SUBTRACT {
                    counts[vw_pos] -= count_change
                } else {
                    counts[vw_pos] += count_change
                }
                
            }
        }
        // edge `{u, w}` (requires `w` in `n_v`)
        for w in n_v.iter() {
            let n_uwv = n_uv.intersection(&graph.neighborhoods[w]);
            let count_change = graph.count_cliques_inside(n_uwv, size - 3);
            if count_change != 0 {
                let uw_pos = Edge::new(u, w).colex_position();
                if SUBTRACT {
                    counts[uw_pos] -= count_change
                } else {
                    counts[uw_pos] += count_change
                }
            }
        }
        if N == 3 {
            return
        }
        // edge `{w, x}` (requires `w, x` in `n_uv`)
        for (w, x) in n_uv.iter().tuple_combinations() {
            let n_uvw = n_uv.intersection(&graph.neighborhoods[w]);
            let n_uvwx = n_uvw.intersection(&graph.neighborhoods[x]);
            let count_change = graph.count_cliques_inside(n_uvwx, size - 4);
            if count_change != 0 {
                let wx_pos = Edge::new(w, x).colex_position();
                if SUBTRACT {
                    counts[wx_pos] -= count_change
                } else {
                    counts[wx_pos] += count_change
                }
            }
        }
    }
}

pub struct ReassignColor {
    edge_pos: usize,
    new_color: usize,
}

impl ReassignColor {
    
}

pub struct CountChange {
    pub old_color: usize,
    pub new_color: usize,
    pub old_count: i32,
    pub new_count: i32,
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
        graphs.iter().zip(counts.iter()).enumerate()
            .map(|(i, (g, c))| format!("g_{i}:\n{g}k_{i}: {c}\n"))
            .join("\n")
    }
}

#[cfg(feature = "tensorboard")]
impl<const C: usize> az_discrete_opt::tensorboard::Summarize for TotalCounts<C> {
    fn summary(&self) -> tensorboard_writer::proto::tensorboard::Summary {
        let mut summary = tensorboard_writer::SummaryBuilder::new();
        for (c, &count) in self.0.iter().enumerate() {
            summary = summary.scalar(&format!("clique_counts/{}", c), count as _);
        }
        summary.build()
    }
}