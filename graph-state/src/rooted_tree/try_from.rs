use super::RootedOrderedTree;

#[derive(Debug)]
pub enum Error {
    InvalidParent,
}

impl<const N: usize> TryFrom<[usize; N]> for RootedOrderedTree<N> {
    type Error = Error;

    fn try_from(parents: [usize; N]) -> Result<Self, Self::Error> {
        for (i, &p) in parents.iter().enumerate().skip(1) {
            if p >= i {
                return Err(Error::InvalidParent);
            }
        }
        Ok(Self { parents })
    }
}
