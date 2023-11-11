use bit_iter::BitIter;
use num_traits::{Unsigned, WrappingShl, PrimInt, WrappingSub};

use super::bitset::Bitset;

// pub trait PrimitiveBitset {
//     type U: Unsigned;
//     fn from_bits(bits: Self::U) -> Self;
// }

// #[derive(Clone)]
// pub struct B64 {
//     bits: u64,
// }

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct PrimitiveBitset<T: Unsigned> {
    bits: T,
}

impl<T: Unsigned> PrimitiveBitset<T> {
    pub const fn from_bits(bits: T) -> Self {
        Self { bits }
    }
}

pub type B8 = PrimitiveBitset<u8>;
pub type B16 = PrimitiveBitset<u16>;
pub type B32 = PrimitiveBitset<u32>;
pub type B64 = PrimitiveBitset<u64>;
pub type B128 = PrimitiveBitset<u128>;

// impl PrimitiveBitset for B64 {
//     type U = u64;
//     fn from_bits(bits: Self::U) -> Self {
//         Self { bits }
//     }
// }

// impl Bitset for B64 {
//     const MAX: u32 = 63;
//     type IterElement = usize;

//     fn empty() -> Self {
//         Self { bits: 0 }
//     }

//     fn singleton_unchecked(n: u32) -> Self {
//         Self { bits: 1 << n }
//     }

//     fn range_to_unchecked(n: u32) -> Self {
//         let bits: u64 = 1 << n;
//         Self { bits: bits.wrapping_sub(1) }
//     }

//     unsafe fn contains_unchecked(&self, n: u32) -> bool {
//         self.bits & (1 << n) != 0
//     }

//     fn cardinality(&self) -> u32 {
//         self.bits.count_ones()
//     }

//     unsafe fn max_unchecked(&self) -> u32 {
//         63 - self.bits.leading_zeros()
//     }

//     fn min_unchecked(&self) -> u32 {
//         self.bits.trailing_zeros()
//     }

//     fn iter(&self) -> impl Iterator<Item = usize> + '_ {
//         BitIter::from(self.bits)
//     }

//     unsafe fn add_unchecked(&mut self, n: u32) {
//         self.bits |= 1 << n;
//     }

//     unsafe fn add_or_remove_unchecked(&mut self, n: u32) {
//         self.bits ^= 1 << n;
//     }

//     fn complement_assign(&mut self) {
//         self.bits = !self.bits;
//     }

//     fn minus_assign(&mut self, other: &Self) {
//         self.bits &= !other.bits;
//     }

//     fn intersection_assign(&mut self, other: &Self) {
//         self.bits &= other.bits;
//     }

//     fn union_assign(&mut self, other: &Self) {
//         self.bits |= other.bits;
//     }

//     fn symmetric_difference_assign(&mut self, other: &Self) {
//         self.bits ^= other.bits;
//     }
// }

trait NumBits {
    const NUM_BITS: u32;
}

impl NumBits for u32 {
    const NUM_BITS: u32 = 32;
}

impl NumBits for u64 {
    const NUM_BITS: u32 = 64;
}

impl<T> Bitset for PrimitiveBitset<T>
where
    T: Unsigned + PrimInt + NumBits + WrappingShl + WrappingSub + core::ops::BitAndAssign
        + core::ops::BitOrAssign + core::ops::BitXorAssign,
    BitIter<T>: From<T> + Iterator<Item = usize>,
{
    const MAX: u32 = T::NUM_BITS - 1;
    type Bits = T;

    // type IterElement = usize;

    fn empty() -> Self {
        Self { bits: T::zero() }
    }

    unsafe fn singleton_unchecked(n: u32) -> Self {
        Self { bits: T::one().wrapping_shl(n) }
    }

    unsafe fn range_to_unchecked(n: u32) -> Self {
        Self { bits: T::one().wrapping_shl(n).wrapping_sub(&T::one()) }
    }

    unsafe fn contains_unchecked(&self, n: u32) -> bool {
        let mut singleton = Self::singleton_unchecked(n);
        singleton.intersection_assign(self);
        !singleton.is_empty()
    }

    fn cardinality(&self) -> u32 {
        self.bits.count_ones()
    }

    unsafe fn max_unchecked(&self) -> u32 {
        Self::MAX.wrapping_sub(self.bits.leading_zeros())
    }

    unsafe fn min_unchecked(&self) -> u32 {
        self.bits.trailing_zeros()
    }

    fn iter(&self) -> impl Iterator<Item = usize> + '_
    where
        Self::Bits: Clone,
    {
        BitIter::from(self.bits.clone())
    }

    unsafe fn add_unchecked(&mut self, n: u32) {
        let singleton = Self::singleton_unchecked(n);
        self.union_assign(&singleton);
    }

    unsafe fn add_or_remove_unchecked(&mut self, n: u32) {
        let singleton = Self::singleton_unchecked(n);
        self.symmetric_difference_assign(&singleton);
    }

    fn complement_assign(&mut self) {
        self.bits = !self.bits;
    }

    fn intersection_assign(&mut self, other: &Self) {
        self.bits.bitand_assign(other.bits);
    }

    fn union_assign(&mut self, other: &Self) {
        self.bits.bitor_assign(other.bits);
    }

    fn symmetric_difference_assign(&mut self, other: &Self) {
        self.bits.bitxor_assign(other.bits);
    }
}