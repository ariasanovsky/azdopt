use core::fmt::Display;

use crate::{simple_graph::edge::Edge, bitset::bitset::Bitset};

use super::{state::Action, BitsetGraph};

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

impl Display for Action {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Action::Add(e) => {
                write!(f, "add {}", e)
            }
            Action::Delete(e) => {
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
