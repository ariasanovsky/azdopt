use core::{num::NonZeroUsize, marker::PhantomData, array::from_fn};
use std::collections::VecDeque;

use bit_iter::BitIter;

use crate::achiche_hansen::{valid::GenericsAreValid, block_forest::BlockForest};

use super::my_bitsets_to_refactor_later::B32;

#[derive(Clone)]
pub struct Neighborhoods<const N: usize> {
    pub(crate) neighborhoods: [B32; N],
}

fn nonzeroes(s: u32) -> Vec<usize> {
    BitIter::from(s).collect()
}

#[test]
fn block_decomposition_of_bowtie() {
    impl Neighborhoods<6> {
        fn bowtie(vertices: [usize; 6]) -> Self {
            let mut neighborhoods = from_fn(|_| B32::empty());
            let [a, b, c, d, e, f] = vertices;
            [(a, b), (b, c), (a, c), (c, d), (d, e), (d, f), (e, f)].into_iter().for_each(|(u, v)| {
                neighborhoods[u].insert_unchecked(v);
                neighborhoods[v].insert_unchecked(u);
            });
            Self::new(neighborhoods)
        }
    }
    let neighborhoods = Neighborhoods::bowtie([0, 4, 5, 1, 2, 3]);
    let block_tree = neighborhoods.block_tree().unwrap();
    todo!("finish writing a test")
}


#[test]
fn block_decomposition_of_c4() {
    impl Neighborhoods<6> {
        fn c4(vertices: [usize; 4]) -> Self {
            let mut neighborhoods = from_fn(|_| B32::empty());
            let [a, b, c, d] = vertices;
            [(a, b), (b, c), (c, d), (a, d)].into_iter().for_each(|(u, v)| {
                neighborhoods[u].insert_unchecked(v);
                neighborhoods[v].insert_unchecked(u);
            });
            Self::new(neighborhoods)
        }
    }
    let neighborhoods = Neighborhoods::c4([0, 1, 2, 3]);
    let block_tree = neighborhoods.block_tree().unwrap();
    todo!("finish writing a test")
}

pub enum Tree {}

impl<const N: usize> Neighborhoods<N> {
    const ALL_VERTICES: B32 = B32::range_to(N);

    pub fn new(neighborhoods: [B32; N]) -> Self {
        let _ = <Self as GenericsAreValid>::VALID;
        Self { neighborhoods }
    }

    pub fn block_tree(&self) -> Option<BlockForest<N, Tree>> {
        let _ = <Self as GenericsAreValid>::VALID;
        
        if N == 0 {
            return Some(BlockForest::new().assert_tree())
        }

        let mut forest = BlockForest::new();
        let mut explored_vertices = B32::empty();
        explored_vertices.insert_unchecked(0);
        forest.insert_root(0);
        
        forest.explore_from(self, &mut explored_vertices, 0);
        if explored_vertices == Self::ALL_VERTICES {
            return Some(forest.assert_tree())
        } else {
            return None
        }
    //     let mut explored_vertices: u32 = 1 << 0;

    //     let mut explored = 1 << 0;
    //     let mut blocks_sets = vec![1 << 0];
    //     let mut blocks_below = vec![1 << 0];
    //     let mut active_blocks: u32 = 1 << 0;

    //     #[derive(Debug)]
    //     enum Insertion {
    //         Root { block_position: usize },
    //         IntoExistingBlock { block_position: usize },
    //         AsNewBlock { parent_position: usize, new_position: usize }, // or `usize` and `NonZeroUsize`
    //     }
    //     impl Insertion {
    //         fn block_position(&self) -> usize {
    //             match self {
    //                 Self::Root { block_position } => *block_position,
    //                 Self::IntoExistingBlock { block_position } => *block_position,
    //                 Self::AsNewBlock { new_position, .. } => *new_position,
    //             }
    //         }
    //     }
    //     let mut insertion_position: [Option<Insertion>; N] = core::array::from_fn(|_| None);
    //     insertion_position[0] = Some(Insertion::Root { block_position: 0 });
    //     let Self { neighborhoods } = self;
    //     neighborhoods.into_iter().enumerate().for_each(|(u, n)| println!("n_{{{u}}} = {:?}", nonzeroes(*n)));
    //     let neighbors = neighborhoods[0];
    //     let mut new_neighborhoods = VecDeque::from([(0usize, neighbors)]);
    //     let mut seen = explored | neighbors;
    //     while let Some((u, n_u)) = new_neighborhoods.pop_front() {
    //         println!("u = {u}");
    //         // println!(
    //         //     "u = {u}:
    //         //     active_blocks = {:?}
    //         //     n_u           = {:?}
    //         //     seen          = {:?}
    //         //     explored      = {:?}",
    //         //     nonzeroes(active_blocks),
    //         //     nonzeroes(n_u),
    //         //     nonzeroes(seen),
    //         //     nonzeroes(explored),
    //         // );
    //         // track the elements in `n_u` to add into `u`'s block
    //         let mut b_u = 0;
    //         // track the explored vertices whose blocks merge with `u`'s block
    //         let mut x_u = 0; // 1 << u;
    //         BitIter::from(n_u).for_each(|v| {
    //             let n_v = neighborhoods[v];
    //             let new_v = !seen & n_v;
    //             if new_v != 0 {
    //                 new_neighborhoods.push_back((v, new_v));
    //                 seen |= new_v;
    //             }
    //             // triangles of the form `{u, v, w}` where `w` is in `n_u` and `n_v`
    //             // if `v` adds `w` to `b_u`, then `w` adds `v` to `b_v` during its iteration
    //             b_u |= n_v & n_u;
    //             // if `v` is adjacent to explored vertices, their blocks merge with `u`'s block
    //             let x_v = explored & n_v;
    //             // if `x_v` is not just `{u}`
    //             if !x_v.is_power_of_two() { // or != 1 << u
    //                 // then add `x_v` to `x_u`
    //                 x_u |= x_v;
    //                 // also add `v` to `b_u`
    //                 b_u |= 1 << v;
    //             }
    //             // println!(
    //             //     "v = {v}:
    //             //     n_v      = {:?}
    //             //     new_v    = {:?}
    //             //     seen     = {:?}
    //             //     b_u      = {:?}
    //             //     x_u      = {:?}
    //             //     x_v      = {:?}",
    //             //     nonzeroes(n_v),
    //             //     nonzeroes(new_v),
    //             //     nonzeroes(seen),
    //             //     nonzeroes(b_u),
    //             //     nonzeroes(x_u),
    //             //     nonzeroes(x_v),
    //             // );
    //         });
    //         x_u ^= 1 << u;
    //         let i_u = insertion_position[u].as_ref().map(|i| i.block_position()).unwrap();
    //         let blocks_below_u = blocks_below[i_u];
    //         let active_blocks_below_u = blocks_below_u & active_blocks;
    //         let j_u = 31 - active_blocks_below_u.leading_zeros() as usize;

    //         // const ZERO_AND_ONE: u32 = 0b11;
    //         // const MIN: u32 = ZERO_AND_ONE.trailing_zeros();
    //         // const MAX: u32 = 31 - ZERO_AND_ONE.leading_zeros();

    //         // println!(
    //         //     "
    //         //     i_u = {i_u}
    //         //     blocks_below_u = {:?}
    //         //     active_blocks_below_u = {:?}
    //         //     j_u = {j_u}",
    //         //     nonzeroes(blocks_below_u),
    //         //     nonzeroes(active_blocks_below_u),
    //         // );
    //         let mut blocks_to_merge: u32 = 1 << j_u;
    //         BitIter::from(x_u).for_each(|v| {
    //             let i_v = insertion_position[v].as_ref().unwrap().block_position();
    //             let j_v = 31 - (blocks_below[i_v] & active_blocks).leading_zeros();
    //             blocks_to_merge |= 1 << j_v;
    //         });
    //         if !blocks_to_merge.is_power_of_two() { // or != 1 << j_u
    //             let mut meet_u = blocks_below[j_u] & active_blocks;
    //             let mut join_u = blocks_below[j_u] & active_blocks;
    //             let blocks_to_merge_minus_u_block = blocks_to_merge ^ (1 << j_u);
    //             BitIter::from(blocks_to_merge_minus_u_block).for_each(|j_v| {
    //                 meet_u &= blocks_below[j_v as usize] & active_blocks;
    //                 join_u |= blocks_below[j_v as usize] & active_blocks;
    //                 blocks_sets[j_u as usize] |= blocks_sets[j_v as usize];
    //             });
    //             let deactivated_blocks = meet_u ^ join_u;
    //             println!(
    //                 "
    //                 meet_u             = {:?}
    //                 join_u             = {:?}
    //                 deactivated_blocks = {:?}",
    //                 nonzeroes(meet_u),
    //                 nonzeroes(join_u),
    //                 nonzeroes(deactivated_blocks),
    //             );
    //             active_blocks ^= deactivated_blocks;
    //             blocks_below[j_u] = join_u;
    //         }
    //         blocks_sets[j_u] |= b_u; // or `^= b_u`
    //         // the remaining vertices form their own blocks (e.g., `uv` is a cut-edge)
    //         let c_u = n_u ^ b_u;
    //         BitIter::from(c_u).for_each(|v| {
    //             let i_v = blocks_sets.len();
    //             let insertion = Insertion::AsNewBlock {
    //                 parent_position: j_u,
    //                 new_position: i_v,
    //             };
    //             // dbg!(u, v, &insertion);
    //             insertion_position[v] = Some(insertion);
    //             blocks_sets.push(1 << v);
    //             let blocks_below_v = blocks_below_u | (1 << i_v);
    //             blocks_below.push(blocks_below_v);
    //             active_blocks |= 1 << i_v;
    //         });
    //         BitIter::from(b_u).for_each(|v| {
    //             let insertion = Insertion::IntoExistingBlock { block_position: j_u };
    //             // dbg!(u, v, &insertion);
    //             insertion_position[v] = Some(insertion);
    //         });
    //         explored |= n_u;
            
    //         BitIter::from(active_blocks).for_each(|j| {
    //             println!("blocks_sets[{}] = {:?}", j, nonzeroes(blocks_sets[j as usize]));
    //         });
    //     }
    //     todo!()
    }
}

impl<const N: usize> Neighborhoods<N> {
    pub fn distance_matrix(&self, blocks: &BlockForest<N>) -> DistanceMatrix<N> {
        todo!()
    }
}

#[derive(Clone)]
pub struct DistanceMatrix<const N: usize> {

}