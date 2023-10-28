use std::{mem::MaybeUninit, collections::BTreeSet};

use az_discrete_opt::{state::Cost, int_min_tree::INTState};
use itertools::Itertools;
use rand::Rng;
use rayon::prelude::{IntoParallelRefMutIterator, IndexedParallelIterator, ParallelIterator};

use crate::{achiche_hansen::my_bitsets_to_refactor_later::B32, ramsey_state::{edge_to_position, edge_from_position}};

use self::{graph::{Neighborhoods, Tree}, block_forest::BlockForest};

mod block_forest;
mod graph;
pub(crate) mod my_bitsets_to_refactor_later;
mod valid;

#[derive(Clone)]
pub struct AHState<const N: usize, const E: usize> {
    neighborhoods: Neighborhoods<N>,
    blocks: BlockForest<N, Tree>,
    edges: [bool; E],
    // add_actions: [bool; E],
    // delete_actions: [bool; E],
    time: usize,
    modified_edges: BTreeSet<usize>,
}

impl<const N: usize, const E: usize> INTState for AHState<N, E> {
    fn act(&mut self, action: usize) {
        let Self {
            neighborhoods,
            edges,
            blocks,
            time,
            modified_edges,
        } = self;
        if action < E {
            // action is an add
            let (u, v) = edge_from_position(action);
            neighborhoods.add_or_delete_edge_unchecked(u, v);
            modified_edges.insert(action);
            edges[action] = true;
        } else {
            // action is a delete
            let (u, v) = edge_from_position(action - E);
            neighborhoods.add_or_delete_edge_unchecked(u, v);
            modified_edges.insert(action - E);
            edges[action - E] = false;
        }
        // todo! lazy approach: recompute the whole thing
        *blocks = neighborhoods.block_tree().unwrap();
        *time -= 1;
    }

    fn is_terminal(&self) -> bool {
        self.time == 0
    }

    fn update_vec(&self, state_vec: &mut [f32]) {
        let Self {
            neighborhoods: _,
            edges,
            blocks,
            time,
            modified_edges,
        } = self;
        let (vec_edges, vec) = state_vec.split_at_mut(E);
        let (vec_add_actions, vec) = vec.split_at_mut(E);
        let (vec_delete_actions, vec_time) = vec.split_at_mut(E);
        assert_eq!(vec_time.len(), 1);
        // initialize edges
        edges.into_iter().zip_eq(vec_edges).for_each(|(e, vec_e)| {
            if *e {
                *vec_e = 1.0;
            } else {
                *vec_e = 0.0;
            }
        });
        // initialize add actions
        vec_add_actions.iter_mut().zip_eq(edges.iter()).for_each(|(vec_a, e)| {
            if *e {
                *vec_a = 0.0;
            } else {
                *vec_a = 1.0;
            }
        });
        // initialize delete actions
        vec_delete_actions.iter_mut().zip_eq(edges.iter()).for_each(|(vec_d, e)| {
            if *e {
                *vec_d = 1.0;
            } else {
                *vec_d = 0.0;
            }
        });
        // prohibit actions on previously modified edges
        modified_edges.iter().for_each(|i| {
            vec_add_actions[*i] = 0.0;
            vec_delete_actions[*i] = 0.0;
        });
        // prohibit delete actions corresponding to cut-edges
        blocks.cut_edges().into_iter().map(|(u, v)| edge_to_position(u, v)).for_each(|i| vec_delete_actions[i] = 0.0);
        // set time
        vec_time[0] = *time as f32;
    }

    fn actions(&self) -> Vec<usize> {
        let Self {
            neighborhoods,
            edges,
            blocks,
            time,
            modified_edges,
        } = self;
        // each missing edge not previous modified is an add action
        let cut_edges = blocks.cut_edges();
        edges.into_iter().enumerate().filter(|(pos, _)| {
            !modified_edges.contains(pos)
        }).filter_map(|(pos, e)| {
            if *e {
                if cut_edges.contains(&edge_from_position(pos)) {
                    None
                } else {
                    Some(pos + E)
                }
            } else {
                Some(pos)
            }
        }).collect()
    }
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
        // dbg!();
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
        
        let mut edges: [bool; E] = [false; E];
        let mut add_actions: [bool; E] = [false; E];
        let mut delete_actions: [bool; E] = [false; E];

        let mut rng = rand::thread_rng();
        let (neighborhoods, blocks) = loop {
            let mut neighborhoods: [B32; N] = core::array::from_fn(|_| B32::empty());
            let edge_iterator = (0..N).flat_map(|v| (0..v).map(move |u| (u, v)));
            edge_iterator
                .zip_eq(edges.iter_mut())
                .zip_eq(add_actions.iter_mut())
                .zip_eq(delete_actions.iter_mut())
            .for_each(|((((u, v), e), a), d)| {
                if rng.gen_bool(p) {
                    *e = true;
                    *a = false;
                    *d = true;
                    neighborhoods[u].insert_unchecked(v);
                    neighborhoods[v].insert_unchecked(u);
                } else {
                    *e = false;
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
            edges,
            blocks,
            time,
            modified_edges: BTreeSet::new(),
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
