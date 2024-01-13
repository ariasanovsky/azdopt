use std::collections::BTreeSet;

use rand::seq::IteratorRandom;

use super::RootedOrderedTree;

#[derive(Clone, Debug)]
pub struct ROTWithActionPermissions<const N: usize> {
    pub tree: RootedOrderedTree<N>,
    pub permitted_actions: BTreeSet<usize>,
}

impl<const N: usize> ROTWithActionPermissions<N> {
    pub fn generate(rng: &mut impl rand::Rng, num_permitted_actions: usize) -> Self {
        let tree = RootedOrderedTree::generate(rng);
        let a = (N - 1) * (N - 2) / 2 - 1;
        let permitted_actions = (0..a).choose_multiple(rng, num_permitted_actions).into_iter().collect();
        Self { tree, permitted_actions }
    }

    pub fn randomize_permitted_actions(&mut self, rng: &mut impl rand::Rng, num_permitted_actions: usize) {
        let a = (N - 1) * (N - 2) / 2 - 1;
        self.permitted_actions = (0..a).choose_multiple(rng, num_permitted_actions).into_iter().collect();
    }
}

impl<const N: usize> core::fmt::Display for ROTWithActionPermissions<N> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{:?}", self.tree)
    }
}