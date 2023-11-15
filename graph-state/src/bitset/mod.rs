// use bit_iter::BitIter;

pub mod bitset;
mod display;
pub mod primitive;

// #[derive(Clone, PartialEq, Eq, Debug)]
// pub struct B32 {
//     bits: u32,
// }

// impl TryFrom<&[usize]> for B32 {
//     type Error = ();

//     fn try_from(value: &[usize]) -> Result<Self, Self::Error> {
//         let mut bits = 0;
//         for &n in value {
//             if n >= 32 {
//                 return Err(());
//             }
//             bits |= 1 << n;
//         }
//         Ok(Self::new(bits))
//     }
// }

// impl B32 {
//     pub const fn empty() -> Self {
//         Self::new(0)
//     }

//     pub const fn is_empty(&self) -> bool {
//         self.bits == 0
//     }

//     pub const fn is_singleton(&self) -> bool {
//         self.bits.is_power_of_two()
//     }

//     pub const fn cardinality(&self) -> u32 {
//         self.bits.count_ones()
//     }

//     pub const fn contains(&self, n: usize) -> bool {
//         self.bits & (1 << n) != 0
//     }

//     pub const fn max_unchecked(&self) -> usize {
//         31 - self.bits.leading_zeros() as usize
//     }

//     pub const fn new(bits: u32) -> Self {
//         Self { bits }
//     }

//     pub const fn singleton_unchecked(u: usize) -> Self {
//         Self::new(1 << u)
//     }

//     pub const fn range_to_unchecked(n: usize) -> Self {
//         assert!(n <= 31);
//         Self::new((1 << n) - 1)
//     }

//     pub const fn minus(&self, other: &Self) -> Self {
//         Self::new(self.bits & !other.bits)
//     }

//     pub const fn intersection(&self, other: &Self) -> Self {
//         Self::new(self.bits & other.bits)
//     }

//     pub const fn symmetric_difference(&self, other: &Self) -> Self {
//         Self::new(self.bits ^ other.bits)
//     }

//     pub fn symmetric_difference_assign(&mut self, other: &Self) {
//         self.bits ^= other.bits
//     }

//     pub fn pop_max(&mut self) -> Option<usize> {
//         if self.is_empty() {
//             None
//         } else {
//             let max = self.max_unchecked();
//             self.add_or_remove_unchecked(max);
//             Some(max)
//         }
//     }

//     pub fn minus_assign(&mut self, other: &Self) {
//         self.bits &= !other.bits;
//     }

//     pub fn add_or_remove_unchecked(&mut self, n: usize) {
//         self.bits ^= 1 << n;
//     }

//     pub fn union_assign(&mut self, other: &Self) {
//         self.bits |= other.bits;
//     }

//     pub fn intersection_assign(&mut self, other: &Self) {
//         self.bits &= other.bits;
//     }

//     pub fn insert_unchecked(&mut self, n: usize) {
//         self.bits |= 1 << n;
//     }

//     pub fn iter(&self) -> impl Iterator<Item = usize> + '_ {
//         BitIter::from(self.bits)
//     }
// }
