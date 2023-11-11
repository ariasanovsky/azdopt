use core::fmt::Display;

use crate::bitset::bitset::Bitset;

impl Display for super::primitive::B32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        self.iter().try_for_each(|n| write!(f, "{}, ", n))?;
        write!(f, "]")
    }
}
