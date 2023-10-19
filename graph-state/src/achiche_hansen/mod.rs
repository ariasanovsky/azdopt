use std::mem::MaybeUninit;

use az_discrete_opt::state::Cost;
use itertools::Itertools;
use rand::Rng;
use rayon::prelude::{IntoParallelRefMutIterator, IndexedParallelIterator, ParallelIterator};

use crate::ramsey_state::edge_to_position;

use self::graph::{BoolEdges, Connected, BlockForest, DistanceMatrix, Neighborhoods};

mod graph;

#[derive(Clone)]
pub struct AHState<const N: usize, const E: usize> {
    // edges: BoolEdges<E, Connected>,
    neighborhoods: Neighborhoods<N, Connected>,
    blocks: BlockForest<N, Connected>,
    add_actions: [bool; E],
    delete_actions: [bool; E],
    distances: DistanceMatrix<N, Connected>,
    time: usize,
}

impl<const N: usize, const E: usize> AHState<N, E> {
    const STATE: usize = 3 * E + 1;
    pub fn par_generate_batch<const BATCH: usize, const STATE: usize>(time: usize, p: f64) -> ([Self; BATCH], [[f32; STATE]; BATCH]) {
        // todo! these asserts are ugly
        let _: () = crate::CheckFirstChooseTwoEqualsSecond::<N, E>::VALID;
        let _: () = crate::CheckFirstTimesThreePlusOneEqualsSecond::<E, STATE>::VALID;
        let mut states: [MaybeUninit<Self>; BATCH] = MaybeUninit::uninit_array();
        let mut vecs: [[f32; STATE]; BATCH] = [[0.0; STATE]; BATCH];
        states.par_iter_mut().zip_eq(vecs.par_iter_mut()).for_each(|(s, v)| {
            s.write(Self::generate(v, time, p));
        });
        let states = unsafe { MaybeUninit::array_assume_init(states) };
        (states, vecs)
    }

    // todo! too many generics
    fn generate<const STATE: usize>(vec: &mut [f32; STATE], time: usize, p: f64) -> Self {
        // todo! improve to comp-time assert, or less ugly
        let _: () = assert!(vec.len() == STATE);
        /* indices 0..3*E are 1.0f32 or 0.0-valued corresponding to:
            0 * E + i => e_i is in E(G)
            1 * E + i => G + e_i is an action
            2 * E + i => G - e_i is an action
            and 3*E is the time as a f32
        */
        let mut rng = rand::thread_rng();
        let (edges, neighborhoods, blocks) = loop {
            let mut edges: [bool; E] = [false; E];
            let mut neighborhoods: [u32; N] = [0; N];
            let edge_iterator = (0..N).flat_map(|v| (0..v).map(move |u| (u, v)));
            edge_iterator.zip_eq(edges.iter_mut()).for_each(|((u, v), e)| {
                if rng.gen_bool(p) {
                    *e = true;
                    neighborhoods[u] |= 1 << v;
                    neighborhoods[v] |= 1 << u;
                }
            });
            let edges = BoolEdges::new(edges);
            let neighborhoods = Neighborhoods::new(neighborhoods);
            if let Some(blocks) = neighborhoods.block_tree() {
                break unsafe {
                    (edges.assert_connected(), neighborhoods.assert_connected(), blocks)
                };
            }
        };
        let add_actions: [bool; E] = edges.complement().edges;
        let mut delete_actions: [bool; E] = [false; E];
        let cut_edges = neighborhoods.cut_edges(&blocks);
        cut_edges.into_iter().for_each(|(u, v)| {
            let pos = edge_to_position(u, v);
            delete_actions[pos] = true;
        });
        let distances = neighborhoods.distance_matrix(&blocks);
        Self {
            // edges,
            neighborhoods,
            blocks,
            add_actions,
            delete_actions,
            distances,
            time,
        }
    }
}

impl<const N: usize, const E: usize> Cost for AHState<N, E> {
    fn cost(&self) -> f32 {
        todo!()
    }
}
