use std::mem::MaybeUninit;

use az_discrete_opt::state::Cost;
use rayon::prelude::{IntoParallelRefMutIterator, IndexedParallelIterator, ParallelIterator};

use self::graph::{BoolEdges, Connected, BlockForest, DistanceMatrix};

mod graph;

#[derive(Clone)]
pub struct AHState<const N: usize, const E: usize> {
    edges: BoolEdges<E, Connected>,
    blocks: BlockForest<N, Connected>,
    add_actions: [bool; E],
    delete_actions: [bool; E],
    distances: DistanceMatrix<N>,
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
        let (edges, blocks) = loop {
            let mut graph = graph::BoolEdges::<E>::generate(&mut rng, p);
            if let Some(g) = graph.to_connected_graph_with_blocks() {
                break g;
            }
        };
        let mut add_actions: [bool; E] = edges.complement().edges;
        let mut delete_actions: [bool; E] = [false; E];
        let distances = todo!();
        Self {
            edges,
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
