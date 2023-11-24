mod try_from;
pub mod space;

/// A rooted ordered tree. For all `i` in `0..N`, the parent of the `i`-th node is in `0..i`.
#[derive(Debug, Clone)]
pub struct RootedOrderedTree<const N: usize> {
    parents: [usize; N],
}

impl<const N: usize> RootedOrderedTree<N> {
    /// Generate a rooted ordered tree. There are `(N-1)!` rooted ordered trees for all `N > 0`.
    pub fn generate(rng: &mut impl rand::Rng) -> Self {
        let mut parents = [0; N];
        for i in 2..N {
            parents[i] = rng.gen_range(0..i);
        }
        Self { parents }
    }

    /// Generate a rooted ordered tree with the constraint that the `N`-th node is a leaf adjacent to the `1`-st.
    /// I.e., the tree is implicitly of the form `N-1` -> `0` <- `1`.
    /// There are `(N-2)!` rooted ordered trees for all `N > 1`.
    pub fn generate_constrained(rng: &mut impl rand::Rng) -> Self {
        let mut parents = [0; N];
        for i in 2..N-1 {
            parents[i] = rng.gen_range(0..i);
        }
        Self { parents }
    }
}
