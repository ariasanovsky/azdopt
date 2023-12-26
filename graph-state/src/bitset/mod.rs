mod display;
pub mod primitive;

#[derive(Debug)]
pub enum Error {
    OutOfBounds,
}

pub trait Bitset {
    const MAX: u32;
    type Bits;
    // type IterElement;
    fn within_bounds(n: u32) -> Result<u32, Error> {
        n.le(&Self::MAX).then_some(n).ok_or(Error::OutOfBounds)
    }
    fn empty() -> Self;
    /// # Safety
    /// `n` must be in 0..32
    unsafe fn singleton_unchecked(n: u32) -> Self;
    fn singleton(n: u32) -> Result<Self, Error>
    where
        Self: Sized,
    {
        Self::within_bounds(n).map(|n| unsafe { Self::singleton_unchecked(n) })
    }
    /// # Safety
    /// `n` must be in 0..32
    unsafe fn range_to_unchecked(n: u32) -> Self;
    fn range_to(n: u32) -> Result<Self, Error>
    where
        Self: Sized,
    {
        Self::within_bounds(n).map(|n| unsafe { Self::range_to_unchecked(n) })
    }
    fn is_empty(&self) -> bool {
        self.cardinality() == 0
    }
    fn is_singleton(&self) -> bool {
        self.cardinality() == 1
    }
    /// # Safety
    /// `n` must be in 0..32
    unsafe fn contains_unchecked(&self, n: u32) -> bool;
    fn contains(&self, n: u32) -> Result<bool, Error> {
        Self::within_bounds(n).map(|n| unsafe { self.contains_unchecked(n) })
    }
    fn cardinality(&self) -> u32;
    /// # Safety
    /// `self` must be non-empty
    unsafe fn max_unchecked(&self) -> u32;
    fn max(&self) -> Option<u32> {
        self.is_empty().then(|| unsafe { self.max_unchecked() })
    }
    /// # Safety
    /// `self` must be non-empty
    unsafe fn min_unchecked(&self) -> u32;
    fn min(&self) -> Option<u32> {
        self.is_empty().then(|| unsafe { self.min_unchecked() })
    }
    fn iter(&self) -> impl Iterator<Item = usize> + '_ + Clone
    where
        Self::Bits: Clone;
    /// # Safety
    /// `n` must be in 0..32
    unsafe fn add_unchecked(&mut self, n: u32);
    fn add(&mut self, n: u32) -> Result<(), Error> {
        Self::within_bounds(n).map(|n| unsafe { self.add_unchecked(n) })
    }
    /// # Safety
    /// `n` must be in 0..32
    unsafe fn add_or_remove_unchecked(&mut self, n: u32);
    fn add_or_remove(&mut self, n: u32) -> Result<(), Error> {
        Self::within_bounds(n).map(|n| unsafe { self.add_or_remove_unchecked(n) })
    }
    fn complement_assign(&mut self);
    fn complement(&self) -> Self
    where
        Self: Sized + Clone,
    {
        let mut result = self.clone();
        result.complement_assign();
        result
    }
    fn minus_assign(&mut self, other: &Self)
    where
        Self: Sized + Clone,
    {
        let other_complement = other.complement();
        self.intersection_assign(&other_complement);
    }
    fn minus(&self, other: &Self) -> Self
    where
        Self: Sized + Clone,
    {
        let mut result = self.clone();
        result.minus_assign(other);
        result
    }
    fn intersection_assign(&mut self, other: &Self);
    fn intersection(&self, other: &Self) -> Self
    where
        Self: Sized + Clone,
    {
        let mut result = self.clone();
        result.intersection_assign(other);
        result
    }
    fn union_assign(&mut self, other: &Self);
    fn union(&self, other: &Self) -> Self
    where
        Self: Sized + Clone,
    {
        let mut result = self.clone();
        result.union_assign(other);
        result
    }
    fn symmetric_difference_assign(&mut self, other: &Self);
    fn symmetric_difference(&self, other: &Self) -> Self
    where
        Self: Sized + Clone,
    {
        let mut result = self.clone();
        result.symmetric_difference_assign(other);
        result
    }

    /// # Safety
    /// `self` must be non-empty
    unsafe fn remove_max_unchecked(&mut self) -> u32 {
        let max = self.max_unchecked();
        self.add_or_remove_unchecked(max);
        max
    }
    fn remove_max(&mut self) -> Option<u32> {
        (!self.is_empty()).then(|| unsafe { self.remove_max_unchecked() })
    }
}
