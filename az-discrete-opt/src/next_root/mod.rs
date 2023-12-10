use crate::int_min_tree::{state_data::StateDataKind, INTMinTree};

impl<P> INTMinTree<P> {
    pub fn unstable_sorted_nodes(&self) -> Vec<(&P, &StateDataKind)> {
        let mut nodes = self
            .data
            .iter()
            .flat_map(|level| level.iter())
            .collect::<Vec<_>>();
        nodes.sort_unstable_by(|a, b| {
            let a_cost = a.1.cost();
            let b_cost = b.1.cost();
            a_cost.partial_cmp(&b_cost).unwrap()
        });
        nodes
    }
}
