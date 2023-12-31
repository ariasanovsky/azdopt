#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(slice_flatten)]
#![feature(sort_floats)]

pub mod bitset;
pub mod ramsey_state;
pub mod rooted_tree;
pub mod simple_graph;
pub mod ramsey_counts;

// todo! we are hard-coding N = 17 to get the R(4, 4) example working

use priority_queue::PriorityQueue;

pub const N: usize = 17;
pub const E: usize = N * (N - 1) / 2;
pub const C: usize = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Color(pub usize);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ColoredCompleteGraph(pub [Color; E]);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MulticoloredGraphEdges(pub [[bool; E]; C]);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MulticoloredGraphNeighborhoods(pub [[u32; N]; C]);

#[derive(Hash, PartialEq, Eq, Clone, Debug)]
pub struct EdgeRecoloring {
    pub new_color: usize,
    pub edge_position: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]

pub struct OrderedEdgeRecolorings(pub PriorityQueue<EdgeRecoloring, i32>);

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CliqueCounts(pub [[i32; E]; 2]);

// #[derive(Debug, Clone)]
// pub struct ColoredGraphWithCounts {
//     pub edges: [[bool; E]; 2],
//     pub neighborhoods: [[u32; N]; 2],
//     pub counts: [[i32; E]; 2],
// }

// impl ColoredGraphWithCounts {
//     pub fn generate_random<R: rand::Rng>(rng: &mut R) -> Self {
//         let mut edges = [[false; E]; 2];
//         let mut neighborhoods = [[0; N]; 2];
//         let pairs = (0..N).map(|j| (0..j).map(move |i| (i, j))).flatten();
//         pairs.enumerate().for_each(|(i, (u, v))| {
//             if rng.gen() {
//                 edges[0][i] = true;
//                 neighborhoods[0][u] |= 1 << v;
//                 neighborhoods[0][v] |= 1 << u;
//             } else {
//                 edges[1][i] = true;
//                 neighborhoods[1][u] |= 1 << v;
//                 neighborhoods[1][v] |= 1 << u;
//             }
//         });
//         let mut counts = [[0; E]; 2];
//         let pairs = (0..N).map(|j| (0..j).map(move |i| (i, j))).flatten();
//         pairs.enumerate().for_each(|(i, (u, v))| {
//             counts[0][i] = Self::count_edges_inside_bitset(neighborhoods[0][u] & neighborhoods[0][v], &neighborhoods[0]);
//             counts[1][i] = Self::count_edges_inside_bitset(neighborhoods[1][u] & neighborhoods[1][v], &neighborhoods[1]);
//         });
//         Self { edges, neighborhoods, counts }
//     }

//     pub fn edges(&self) -> &[[bool; E]; 2] {
//         &self.edges
//     }

//     pub fn counts(&self) -> &[[i32; E]; 2] {
//         &self.counts
//     }

//     fn count_edges_inside_bitset(bitset: u32, neighborhoods: &[u32; N]) -> i32 {
//         let count = (0..N).filter(|w| bitset & (1 << w) != 0).map(|w| bitset & neighborhoods[w]).map(|neigh| neigh.count_ones()).sum::<u32>() / 2;
//         count as i32
//     }
// }

// #[test]
// fn naively_count() {
//     let g = ColoredGraphWithCounts::generate_random(&mut rand::thread_rng());
//     let pairs = (0..N).map(|j| (0..j).map(move |i| (i, j))).flatten();
//     pairs.enumerate().for_each(|(i, (u, v))| {
//         let pairs = (0..N).map(|j| (0..j).map(move |i| (i, j))).flatten();
//         let red_uvwx = pairs.filter(|(w, x)| {
//             // check each of the 5 edges (u, w), (u, x), (v, w), (v, x), (w, x) with the neighborhood bitset
//             let bitset = g.neighborhoods[0][u] & g.neighborhoods[0][v];
//             (bitset & (1 << w) != 0) && (bitset & (1 << x) != 0) && g.neighborhoods[0][*w] & (1 << x) != 0
//         }).count();
//         debug_assert_eq!(red_uvwx, g.counts[0][i] as usize);
//         let pairs = (0..N).map(|j| (0..j).map(move |i| (i, j))).flatten();
//         let blue_uvwx = pairs.filter(|(w, x)| {
//             // check each of the 5 edges (u, w), (u, x), (v, w), (v, x), (w, x) with the neighborhood bitset
//             let bitset = g.neighborhoods[1][u] & g.neighborhoods[1][v];
//             (bitset & (1 << w) != 0) && (bitset & (1 << x) != 0) && g.neighborhoods[1][*w] & (1 << x) != 0
//         }).count();
//         debug_assert_eq!(blue_uvwx, g.counts[1][i] as usize);
//     });
// }

// borrorwed from faer: https://github.com/sarah-ek/pulp/blob/17902ba463667a02e21a509b1b94ccbf62c4e75f/src/lib.rs#L1297-L1346
// #[doc(hidden)]
// pub struct CheckFirstChooseTwoEqualsSecond<const N: usize, const E: usize>(core::marker::PhantomData<([(); N], [(); E])>);
// impl<const N: usize, const E: usize> CheckFirstChooseTwoEqualsSecond<N, E> {
//     pub const VALID: () = {
//         debug_assert!(N * (N - 1) / 2 == E);
//     };
// }

// #[macro_export]
// macro_rules! static_assert_first_choose_two_equals_second {
//     ($n: expr, $e: expr) => {
//         const _: () = $crate::CheckFirstChooseTwoEqualsSecond::<$n, $e>::VALID;
//     };
// }

// #[doc(hidden)]
// pub struct CheckFirstTimesThreePlusOneEqualsSecond<const E: usize, const STATE: usize>(core::marker::PhantomData<([(); E], [(); STATE])>);
// impl<const E: usize, const STATE: usize> CheckFirstTimesThreePlusOneEqualsSecond<E, STATE> {
//     pub const VALID: () = {
//         debug_assert!(3 * E + 1 == STATE);
//     };
// }

// #[macro_export]
// macro_rules! static_assert_first_times_three_plus_one_equals_second {
//     ($n: expr, $s: expr) => {
//         const _: () = $crate::CheckFirstChooseTwoEqualsSecond::<$n, $s>::VALID;
//     };
// }
