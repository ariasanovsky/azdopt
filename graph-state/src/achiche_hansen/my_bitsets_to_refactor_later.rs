use bit_iter::BitIter;

#[derive(Clone, PartialEq, Eq)]
pub(crate) struct B32 {
    bits: u32,
}

impl B32 {
    pub const fn new(bits: u32) -> Self {
        Self { bits }
    }

    pub const fn empty() -> Self {
        Self::new(0)
    }

    pub const fn range_to(n: usize) -> Self {
        let _: () = assert!(n <= 31);
        Self::new((1 << n) - 1)
    }

    pub const fn minus(&self, other: &Self) -> Self {
        Self::new(self.bits & !other.bits)
    }

    pub const fn is_empty(&self) -> bool {
        self.bits == 0
    }

    pub const fn intersection(&self, other: &Self) -> Self {
        Self::new(self.bits & other.bits)
    }

    pub const fn is_singleton(&self) -> bool {
        self.bits.is_power_of_two()
    }

    pub const fn symmetric_difference(&self, other: &Self) -> Self {
        Self::new(self.bits ^ other.bits)
    }

    pub fn max_unchecked(&self) -> usize {
        dbg!(self.bits);
        31 - self.bits.leading_zeros() as usize
    }

    pub fn minus_assign(&mut self, other: &Self) {
        self.bits &= !other.bits;
    }

    pub fn add_or_remove_unchecked(&mut self, n: usize) {
        self.bits ^= 1 << n;
    }

    pub fn union_assign(&mut self, other: &Self) {
        self.bits |= other.bits;
    }

    pub fn intersection_assign(&mut self, other: &Self) {
        self.bits &= other.bits;
    }

    pub fn insert_unchecked(&mut self, n: usize) {
        self.bits |= 1 << n;
    }

    pub fn iter(&self) -> BitIter<u32> {
        BitIter::from(self.bits)
    }
}