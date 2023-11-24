use faer::Faer;

use crate::simple_graph::{edge::Edge, connected_bitset_graph::Conjecture2Dot1Cost};

use super::RootedOrderedTree;

/// An ordered edge.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct OrderedEdge {
    edge: Edge,
}

impl OrderedEdge {
    pub fn new(edge: Edge) -> Self {
        Self { edge }
    }

    pub fn edge(&self) -> &Edge {
        &self.edge
    }

    pub fn parent(&self) -> usize {
        self.edge().min()
    }

    pub fn child(&self) -> usize {
        self.edge().max()
    }

    pub fn child_parent(&self) -> (usize, usize) {
        self.edge().vertices()
    }

    /// This method will panic if the edge is `0 <- 1`.
    pub fn index_ignoring_edge_0_1(&self) -> usize {
        debug_assert!(self.edge() != &Edge::new(0, 1));
        self.edge().colex_position() - 1
    }

    pub fn from_index_ignoring_edge_0_1(index: usize) -> Self {
        Self::new(Edge::from_colex_position(index + 1))
    }
}

impl<const N: usize> RootedOrderedTree<N> {
    pub fn set_parent(&mut self, ordered_edge: &OrderedEdge) {
        let parents = self.parents_mut();
        let (max, min) = ordered_edge.edge.vertices();
        parents[max] = min;
    }

    pub fn possible_parent_modifications(&self, child: usize) -> impl Iterator<Item = OrderedEdge> + '_ {
        self.parent(child).map(|parent| {
            (0..parent).chain(parent + 1..child).map(move |new_parent| {
                OrderedEdge::new(Edge::new(new_parent, child))
            })
        }).into_iter().flatten()
    }

    pub fn all_possible_parent_modifications(&self) -> impl Iterator<Item = OrderedEdge> + '_ {
        (0..N).map(|child| self.possible_parent_modifications(child)).flatten()
    }

    pub fn conjecture_2_1_cost(&self) -> Conjecture2Dot1Cost {
        let a = self.adjacency_matrix();
        let lambda_1 = a.selfadjoint_eigenvalues(faer::Side::Upper).into_iter().max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap(); 
        assert!(
            lambda_1 >= 1.4,
            "lambda_1: {lambda_1}, a:\n{a:?}",
        );
        let matching = self.maximum_matching();
        Conjecture2Dot1Cost { lambda_1, matching }
    }

    pub fn adjacency_matrix(&self) -> faer::Mat<f64> {
        let mut a = faer::Mat::zeros(N, N);
        for (parent, child) in self.parents().iter().enumerate().skip(1) {
            a[(parent, *child)] = 1.0;
            a[(*child, parent)] = 1.0;
        }
        a
    }

    /// unoptimized
    pub fn maximum_matching(&self) -> Vec<Edge> {
        let mut m = vec![];
        let mut available = [true; N];
        loop {
            // find the next leaves to remove
            let mut next_leaf = available.clone();
            for i in 1..N {
                if available[i] {
                    let parent = self.parent(i).unwrap();
                    next_leaf[parent] = false;
                }
            }
            // remove each leaf and its parent
            for i in 1..N {
                if next_leaf[i] {
                    available[i] = false;
                    let parent = self.parent(i).unwrap();
                    if available[parent] {
                        available[parent] = false;
                        m.push(Edge::new(parent, i));
                    }
                }
            }
            // break if there are fewer than 2 vertices left
            let num_available = available.iter().filter(|&&a| a).count();
            if num_available < 2 {
                break;
            }
        }
        m
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use crate::{rooted_tree::RootedOrderedTree, simple_graph::{edge::Edge, connected_bitset_graph::Conjecture2Dot1Cost}};

    use super::OrderedEdge;

    #[test]
    fn oredered_edges_on_first_five_vertices_have_correct_index() {
        let expected_edge = [(0, 2), (1, 2), (0, 3), (1, 3), (2, 3), (0, 4), (1, 4), (2, 4), (3, 4)].map(|(parent, child)| {
            OrderedEdge::new(Edge::new(parent, child))
        });
        for expected_index in 0..9 {
            let edge = OrderedEdge::from_index_ignoring_edge_0_1(expected_index);
            let expected_edge = &expected_edge[expected_index];
            let index = edge.index_ignoring_edge_0_1();
            assert!(
                index == expected_index && &edge == expected_edge,
                "edge: {edge:?}, expected_edge: {expected_edge:?}, index: {index}, expected_index: {expected_index}",
            );
        }
    }

    #[test]
    fn star_on_five_vertices_has_correct_possible_parent_modifications() {
        let star = RootedOrderedTree::try_from([0, 0, 0, 0, 0]).unwrap();
        let expected = [(2, 1), (3, 1), (3, 2), (4, 1), (4, 2), (4, 3)].map(|(parent, child)| {
            OrderedEdge::new(Edge::new(parent, child))
        });
        let actual = star.all_possible_parent_modifications().collect::<Vec<_>>();
        assert_eq!(actual, expected);
    }

    #[test]
    fn generate_constrained_generates_all_trees_on_five_vertices() {
        let mut rng = rand::thread_rng();
        let mut trees = BTreeSet::new();
        let expected_tree_parents = [
            [0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 2, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 0],
            [0, 0, 1, 2, 0],
        ];
        for _ in 0..10_000 {
            let tree = RootedOrderedTree::<5>::generate_constrained(&mut rng);
            trees.insert(tree.parents().clone());
            if trees.len() == 6 {
                break;
            }
        }
        let actual_tree_parents = trees.into_iter().collect::<Vec<_>>();
        assert_eq!(actual_tree_parents, expected_tree_parents);
    }

    #[test]
    fn star_on_five_vertices_has_correct_conjecture_2_1_cost() {
        let star = RootedOrderedTree::try_from([0, 0, 0, 0, 0]).unwrap();
        let expected_lambda_1 = 2.0;
        let possible_expected_matchings = vec![
            vec![Edge::new(0, 1)],
            vec![Edge::new(0, 2)],
            vec![Edge::new(0, 3)],
            vec![Edge::new(0, 4)],
        ];
        let Conjecture2Dot1Cost {
            lambda_1: actual_lambda_1,
            matching: actual_matching,
        } = star.conjecture_2_1_cost();
        assert!((actual_lambda_1 - expected_lambda_1).abs() < 1e-6);
        assert!(possible_expected_matchings.contains(&actual_matching));
    }

    #[test]
    fn path_graph_on_five_vertices_has_correct_conjecture_2_1_cost() {
        let path_graph = RootedOrderedTree::try_from([0, 0, 1, 2, 3]).unwrap();
        let expected_lambda_1 = 2.0 * f64::cos(std::f64::consts::PI / 6.0);
        let possible_expected_matchings = vec![
            vec![Edge::new(0, 1), Edge::new(2, 3)],
            vec![Edge::new(0, 1), Edge::new(3, 4)],
            vec![Edge::new(1, 2), Edge::new(3, 4)],
        ];
        let Conjecture2Dot1Cost {
            lambda_1: actual_lambda_1,
            matching: mut actual_matching,
        } = path_graph.conjecture_2_1_cost();
        actual_matching.sort();
        assert!((actual_lambda_1 - expected_lambda_1).abs() < 1e-6);
        assert!(
            possible_expected_matchings.contains(&actual_matching),
            "actual_matching: {actual_matching:?}",
        );
    }
}
