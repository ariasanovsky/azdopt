use crate::int_min_tree::{INTMinTree, NewTreeLevel};

pub struct TreeData<const BATCH: usize, P> {
    trees: [INTMinTree<P>; BATCH],
    paths: [P; BATCH],
    nodes: [Option<NewTreeLevel<P>>; BATCH],
}

impl<const BATCH: usize, P> TreeData<BATCH, P> {
    
}