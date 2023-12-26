use core::fmt::Display;

use crate::{bitset::Bitset, simple_graph::edge::Edge};

use super::{space::action::AddOrDeleteEdge, BitsetGraph};

impl<const N: usize, B: Bitset> Display for BitsetGraph<N, B>
where
    B::Bits: Clone,
{
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { neighborhoods } = self;
        neighborhoods
            .iter()
            .enumerate()
            .try_for_each(|(i, neighborhood)| {
                write!(f, "node {}:", i)?;
                neighborhood
                    .iter()
                    .try_for_each(|edge| write!(f, "  {}", edge))?;
                writeln!(f)
            })
    }
}

impl Display for AddOrDeleteEdge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AddOrDeleteEdge::Add(e) => {
                write!(f, "add {}", e)
            }
            AddOrDeleteEdge::Delete(e) => {
                write!(f, "delete {}", e)
            }
        }
    }
}

impl Display for Edge {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { max, min } = self;
        write!(f, "[{}, {}]", min, max)
    }
}
