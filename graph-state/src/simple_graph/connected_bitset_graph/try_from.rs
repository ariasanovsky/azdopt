use super::ConnectedBitsetGraph;

impl<const N: usize> TryFrom<&[(usize, usize)]> for ConnectedBitsetGraph<N> {
    type Error = ();

    fn try_from(value: &[(usize, usize)]) -> Result<Self, Self::Error> {
        let graph = crate::simple_graph::bitset_graph::BitsetGraph::<N>::try_from(value)?;
        graph.to_connected().ok_or(())
    }
}
