use core::fmt::Display;

use crate::{bitset::Bitset, simple_graph::edge::Edge};

use super::{space::action::AddOrDeleteEdge, BitsetGraph};

impl<const N: usize> Display for BitsetGraph<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { neighborhoods } = self;
        neighborhoods
            .iter()
            .enumerate()
            .try_for_each(|(i, neighborhood)| {
                writeln!(f, "node {}:", i)?;
                neighborhood
                    .iter()
                    .try_for_each(|edge| writeln!(f, "  {}", edge))
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
