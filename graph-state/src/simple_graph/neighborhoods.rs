use crate::bitset::B32;

// todo! delete?
#[derive(Clone)]
pub struct Neighborhoods<const N: usize> {
    pub(crate) neighborhoods: [B32; N],
}
