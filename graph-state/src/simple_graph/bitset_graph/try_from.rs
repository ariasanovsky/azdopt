use crate::bitset::B32;

use super::BitsetGraph;

impl<const N: usize> TryFrom<&[(usize, usize)]> for BitsetGraph<N> {
    type Error = ();

    fn try_from(value: &[(usize, usize)]) -> Result<Self, Self::Error> {
        let mut neighborhoods = core::array::from_fn(|_| B32::empty());
        for &(v, u) in value {
            if v >= N || u >= N || v == u {
                return Err(());
            }
            neighborhoods[v].add_or_remove_unchecked(u);
            neighborhoods[u].add_or_remove_unchecked(v);
        }
        Ok(Self { neighborhoods })
    } 
}