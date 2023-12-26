use core::fmt::Display;

use crate::bitset::Bitset;

use super::ConnectedBitsetGraph;

impl<const N: usize, B> Display for ConnectedBitsetGraph<N, B>
where
    B: Bitset,
    B::Bits: Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { neighborhoods } = self;
        neighborhoods
            .iter()
            .enumerate()
            .try_for_each(|(u, neighborhood)| {
                write!(f, "n_{{{u}}} = [")?;
                neighborhood.iter().try_for_each(|v| write!(f, "{v}, "))?;
                writeln!(f, " ]")
            })
    }
}
