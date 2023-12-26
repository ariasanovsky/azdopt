use crate::bitset::Bitset;

use super::ConnectedBitsetGraph;

impl<const N: usize, B> TryFrom<&[(usize, usize)]> for ConnectedBitsetGraph<N, B>
where
    B: Bitset + Clone + PartialEq,
    B::Bits: Clone,
{
    type Error = ();

    fn try_from(value: &[(usize, usize)]) -> Result<Self, Self::Error> {
        let graph = crate::simple_graph::bitset_graph::BitsetGraph::<N, B>::try_from(value)?;
        graph.to_connected().ok_or(())
    }
}
