impl<const N: usize> core::fmt::Display for super::PrueferCode<N> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        format!("{:?}", self.code()).fmt(f)
    }
}
