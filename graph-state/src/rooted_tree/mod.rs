pub mod modify_parent_once;
pub mod ordered_edge;
pub mod space;
mod try_from;

/// A rooted ordered tree. For all `i` in `0..N`, the parent of the `i`-th node is in `0..i`.
#[derive(Debug, Clone)]
pub struct RootedOrderedTree<const N: usize> {
    parents: [usize; N],
}

impl<const N: usize> RootedOrderedTree<N> {
    /// Generate a rooted ordered tree. There are `(N-1)!` rooted ordered trees for all `N > 0`.
    pub fn generate(rng: &mut impl rand::Rng) -> Self {
        let mut parents = [0; N];
        for (i, parent) in parents.iter_mut().enumerate().take(N - 1).skip(2) {
            *parent = rng.gen_range(0..i);
        }
        Self { parents }
    }

    /// Generate a rooted ordered tree with the constraint that the `N`-th node is a leaf adjacent to the `1`-st.
    /// I.e., the tree is implicitly of the form `N-1` -> `0` <- `1`.
    /// There are `(N-2)!` rooted ordered trees for all `N > 1`.
    pub fn generate_constrained(rng: &mut impl rand::Rng) -> Self {
        let mut parents = [0; N];
        for (i, parent) in parents.iter_mut().enumerate().take(N - 1).skip(2) {
            *parent = rng.gen_range(0..i);
        }
        Self { parents }
    }

    pub(crate) fn parents(&self) -> &[usize; N] {
        &self.parents
    }

    pub(crate) fn parents_ignoring_last_vertex(&self) -> &[usize] {
        &self.parents[..N - 1]
    }

    pub(crate) fn parents_mut(&mut self) -> &mut [usize; N] {
        &mut self.parents
    }

    fn parent(&self, child: usize) -> Option<usize> {
        if child == 0 {
            None
        } else {
            Some(self.parents()[child])
        }
    }

    // pub fn edges(&self) -> impl Iterator<Item = ordered_edge::OrderedEdge> + '_ {
    //     self.parents().iter().enumerate().skip(1).map(|(child, &parent)| {
    //         debug_assert!(parent < child);
    //         ordered_edge::OrderedEdge::new(crate::simple_graph::edge::Edge::new(parent, child))
    //     })
    // }

    pub fn edge_indices_ignoring_0_1_and_last_vertex(&self) -> impl Iterator<Item = usize> + '_ {
        self.parents_ignoring_last_vertex()
            .iter()
            .enumerate()
            .skip(2)
            .map(|(child, &parent)| {
                debug_assert!(parent < child);
                let edge = ordered_edge::OrderedEdge::new(crate::simple_graph::edge::Edge::new(
                    parent, child,
                ));
                edge.index_ignoring_edge_0_1()
            })
    }
}
