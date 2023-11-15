use std::collections::BTreeMap;

pub struct ActionMultiset {
    pub(crate) actions: BTreeMap<usize, usize>,
}
