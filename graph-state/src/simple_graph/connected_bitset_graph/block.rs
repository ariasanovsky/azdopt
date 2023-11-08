#[cfg(test)]
mod tests {
    use crate::simple_graph::{connected_bitset_graph::ConnectedBitsetGraph, edge::Edge};

    #[test]
    fn cycle_on_four_vertices_has_no_cut_edges() {
        let c4: ConnectedBitsetGraph::<4> = [(0, 1), (1, 2), (2, 3), (3, 0)].as_ref().try_into().unwrap();
        let cuts = c4.cut_edges().collect::<Vec<_>>();
        assert_eq!(
            &cuts,
            &[]
        )
    }

    #[test]
    fn paw_graph_has_one_cut_edge() {
        let paw: ConnectedBitsetGraph::<4> = [(0, 1), (1, 2), (1, 3), (2, 3)].as_ref().try_into().unwrap();
        let cuts = paw.cut_edges().collect::<Vec<_>>();
        assert_eq!(
            &cuts,
            &[(0, 1)].map(|(u, v)| Edge::new(u, v))
        )
    }
}