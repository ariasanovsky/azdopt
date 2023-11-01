use core::fmt::Display;

use super::B32;

impl Display for B32 {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[")?;
        self.iter().try_for_each(|n| write!(f, "{}, ", n))?;
        write!(f, "]")
    }
}
