use crate::bitset::{primitive::B32, Bitset};

use super::BitsetGraph;

impl<const N: usize> TryFrom<&[(usize, usize)]> for BitsetGraph<N> {
    type Error = ();

    fn try_from(value: &[(usize, usize)]) -> Result<Self, Self::Error> {
        let mut neighborhoods = core::array::from_fn(|_| B32::empty());
        for &(v, u) in value {
            if v >= N || u >= N || v == u {
                return Err(());
            }
            unsafe { neighborhoods[v].add_or_remove_unchecked(u as u32) };
            unsafe { neighborhoods[u].add_or_remove_unchecked(v as u32) };
        }
        Ok(Self { neighborhoods })
    }
}
