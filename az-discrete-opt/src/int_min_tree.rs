use std::collections::BTreeMap;

use crate::iq_min_tree::ActionsTaken;

pub struct INTMinTree {
    root_data: INTStateData,
    data: BTreeMap<ActionsTaken, INTStateData>,
}

impl INTMinTree {
    pub fn new_forest<const BATCH: usize>() -> [Self; BATCH] {
        core::array::from_fn(|_| Self::new())
    }

    pub fn new() -> Self {
        Self {
            root_data: INTStateData::new(),
            data: BTreeMap::new(),
        }
    }
}

struct INTStateData;

impl INTStateData {
    pub fn new() -> Self {
        todo!()
    }
}
