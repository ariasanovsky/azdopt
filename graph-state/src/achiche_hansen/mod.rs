use std::mem::MaybeUninit;

use az_discrete_opt::state::Cost;
use itertools::Itertools;
use rand::Rng;
use rayon::prelude::{IntoParallelRefMutIterator, IndexedParallelIterator, ParallelIterator};

use crate::{achiche_hansen::my_bitsets_to_refactor_later::B32, ramsey_state::edge_to_position};

use self::{graph::{Neighborhoods, Tree}, block_forest::BlockForest};

mod block_forest;
mod graph;
pub(crate) mod my_bitsets_to_refactor_later;
mod valid;

#[derive(Clone)]
pub struct AHState<const N: usize, const E: usize> {
    neighborhoods: Neighborhoods<N>,
    blocks: BlockForest<N, Tree>,
    add_actions: [bool; E],
    delete_actions: [bool; E],
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
        dbg!();
        // todo! improve to comp-time assert, or less ugly
        let _: () = assert!(vec.len() == Self::STATE);
        /* indices 0..3*E are 1.0f32 or 0.0-valued corresponding to:
            0 * E + i => e_i is in E(G)
            1 * E + i => G + e_i is an action
            2 * E + i => G - e_i is an action
            and 3*E is the time as a f32
        */
        let (vec_edges, vec) = vec.split_at_mut(E);
        let (vec_add_actions, vec) = vec.split_at_mut(E);
        let (vec_delete_actions, vec_time) = vec.split_at_mut(E);
        assert_eq!(vec_time.len(), 1);
        
        let mut add_actions: [bool; E] = [false; E];
        let mut delete_actions: [bool; E] = [false; E];

        let mut rng = rand::thread_rng();
        let (neighborhoods, blocks) = loop {
            let mut neighborhoods: [B32; N] = core::array::from_fn(|_| B32::empty());
            let edge_iterator = (0..N).flat_map(|v| (0..v).map(move |u| (u, v)));
            edge_iterator
                .zip_eq(add_actions.iter_mut())
                .zip_eq(delete_actions.iter_mut())
            .for_each(|(((u, v), a), d)| {
                if rng.gen_bool(p) {
                    *a = false;
                    *d = true;
                    neighborhoods[u].insert_unchecked(v);
                    neighborhoods[v].insert_unchecked(u);
                } else {
                    *a = true;
                    *d = false;
                }
            });
            let neighborhoods = Neighborhoods::new(neighborhoods);
            if let Some(blocks) = neighborhoods.block_tree() {
                // delete the cut-edges from the set of delete actions
                blocks.cut_edges().into_iter().map(|(u, v)| edge_to_position(u, v)).for_each(|i| delete_actions[i] = false);
                break (neighborhoods, blocks)
            }
        };

        // indicate edges with 1.0 (iff the add action is not valid) or 0.0
        vec_edges.iter_mut().zip_eq(vec_add_actions).zip_eq(add_actions.iter()).for_each(|((vec_e, vec_a), a)| {
            if *a {
                *vec_e = 0.0;
                *vec_a = 1.0;
            } else {
                *vec_e = 1.0;
                *vec_a = 0.0;
            }
        });
        vec_delete_actions.iter_mut().zip_eq(delete_actions.iter()).for_each(|(vec_d, d)| {
            if *d {
                *vec_d = 1.0;
            } else {
                *vec_d = 0.0;
            }
        });
        vec_time[0] = time as f32;
        Self {
            neighborhoods,
            blocks,
            add_actions,
            delete_actions,
            time,
        }
    }
}

impl<const N: usize, const E: usize> Cost for AHState<N, E> {
    fn cost(&self) -> f32 {
        let distance_matrix = self.neighborhoods.distance_matrix(self.blocks.forget_state());
        let eigs = distance_matrix.eigenvalues();
        let k = (2 * N) / 3 - 1;
        let proximity = distance_matrix.proximity();
        (eigs[k] + proximity) as f32
    }
}
