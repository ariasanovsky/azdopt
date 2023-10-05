use core::mem::MaybeUninit;
use core::{mem::transmute, array::from_fn};

use bit_iter::BitIter;
use priority_queue::PriorityQueue;
use ramsey::{ColoredCompleteGraph, MulticoloredGraphEdges, MulticoloredGraphNeighborhoods, OrderedEdgeRecolorings, CliqueCounts, C, E, Color, N, EdgeRecoloring};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};

const ACTION: usize = C * E;
const BATCH: usize = 64;

#[derive(Debug)]
struct GraphState {
    colors: ColoredCompleteGraph,
    edges: MulticoloredGraphEdges,
    neighborhoods: MulticoloredGraphNeighborhoods,
    available_actions: [[bool; E]; C],
    ordered_actions: OrderedEdgeRecolorings,
    counts: CliqueCounts,
    time_remaining: usize,
}

impl GraphState {
    fn generate_random<R: rand::Rng>(t: usize, rng: &mut R) -> Self {
        let mut edges: [[bool; E]; C] = [[false; E]; C];
        let mut neighborhoods: [[u32; N]; C] = [[0; N]; C];
        let mut colors: [MaybeUninit<Color>; E] = unsafe {
            let colors: MaybeUninit<[Color; E]> = MaybeUninit::uninit();
            transmute(colors)
        };
        let mut available_actions: [[bool; E]; C] = [[true; E]; C];
        let edge_iterator = (0..N).map(|v| (0..v).map(move |u| (u, v))).flatten();
        edge_iterator.zip(colors.iter_mut()).enumerate().for_each(|(i, ((u, v), color))| {
            let c = rng.gen_range(0..C);
            edges[c][i] = true;
            available_actions[c][i] = false;
            neighborhoods[c][u] |= 1 << v;
            neighborhoods[c][v] |= 1 << u;
            color.write(Color(c));
        });
        let colors: [Color; E] = unsafe {
            transmute(colors)
        };
        let mut counts: [[MaybeUninit<i32>; E]; C] = unsafe {
            let counts: MaybeUninit<[[i32; E]; C]> = MaybeUninit::uninit();
            transmute(counts)
        };
        neighborhoods.iter().zip(counts.iter_mut()).for_each(|(neighborhoods, counts)| {
            let edge_iterator = (0..N).map(|v| (0..v).map(move |u| (u, v))).flatten();
            edge_iterator.zip(counts.iter_mut()).for_each(|((u, v), k)| {
                let neighborhood = neighborhoods[u] & neighborhoods[v];
                let count = BitIter::from(neighborhood).map(|w| {
                    (neighborhood & neighborhoods[w]).count_ones()
                }).sum::<u32>() / 2;
                k.write(count as i32);
            });
        });
        let counts: [[i32; E]; C] = unsafe {
            transmute(counts)
        };
        let mut recolorings: PriorityQueue<EdgeRecoloring, i32> = PriorityQueue::new();
        colors.iter().enumerate().for_each(|(i, c)| {
            let old_color = c.0;
            let old_count = counts[old_color][i];
            (0..C).filter(|c| old_color.ne(c)).for_each(|new_color| {
                let new_count = counts[new_color][i];
                let reward = old_count - new_count;
                let recoloring = EdgeRecoloring { new_color, edge_position: i };
                recolorings.push(recoloring, reward);
            })
        });
        Self {
            colors: ColoredCompleteGraph(colors),
            edges: MulticoloredGraphEdges(edges),
            neighborhoods: MulticoloredGraphNeighborhoods(neighborhoods),
            available_actions,
            ordered_actions: OrderedEdgeRecolorings(recolorings),
            counts: CliqueCounts(counts),
            time_remaining: t,
        }
    }

    fn generate_batch(t: usize) -> [Self; BATCH] {
        let mut graphs: [MaybeUninit<Self>; BATCH] = unsafe {
            MaybeUninit::uninit().assume_init()
        };
        graphs.par_iter_mut().for_each(|g| {
            g.write(GraphState::generate_random(t, &mut rand::thread_rng()));
        });
        unsafe {
            transmute(graphs)
        }
    }
}



fn main() {
    let graphs: [GraphState; BATCH] = GraphState::generate_batch(20);
    dbg!(&graphs[0]);
}