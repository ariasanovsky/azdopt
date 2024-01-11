use crate::bitset::Bitset;

use super::{RamseyCounts, no_recolor::RamseyCountsNoRecolor};

impl<const N: usize, const E: usize, const C: usize, B> core::fmt::Display for RamseyCountsNoRecolor<N, E, C, B>
where
    B: Bitset,
    B::Bits: Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.state.fmt(f)
    }
}

impl<const N: usize, const E: usize, const C: usize, B> core::fmt::Display for RamseyCounts<N, E, C, B>
where
    B: Bitset,
    B::Bits: Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        (0..N).try_for_each(|u| {
            write!(f, "{u:3}:\t")?;
            self.graph.graphs.iter().map(|g| &g.neighborhoods[u]).try_for_each(|n| {
                write!(f, "{{")?;
                n.iter().try_for_each(|v| write!(f, "{v:3}, "))?;
                write!(f, "}}, ")
            })?;
            writeln!(f)?;
            Ok(())
        })
    }
}
