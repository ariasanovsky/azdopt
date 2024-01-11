#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]
#![feature(slice_flatten)]
#![feature(sort_floats)]

pub mod bitset;
pub mod ramsey_counts;
pub mod ramsey_state;
pub mod rooted_tree;
pub mod simple_graph;

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
