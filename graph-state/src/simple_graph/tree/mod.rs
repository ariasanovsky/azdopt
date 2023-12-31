use std::collections::VecDeque;

use faer::Faer;

use self::space::action::PrueferCodeEntry;

use super::{connected_bitset_graph::Conjecture2Dot1Cost, edge::Edge};

mod display;
pub mod space;
pub mod sparse6;

#[derive(Debug, Clone)]
pub struct PrueferCode<const N: usize> {
    code: [usize; N], // no const generics, else we'd put N - 2 here
}

impl<const N: usize> PrueferCode<N> {
    pub fn generate(rng: &mut impl rand::Rng) -> Self {
        let mut code: [usize; N] = [0; N];
        (0..(N - 2)).for_each(|i| code[i] = rng.gen_range(0..N));
        Self { code }
    }

    pub fn code(&self) -> &[usize] {
        &self.code[..(N - 2)]
    }
    pub fn entries(&self) -> impl Iterator<Item = PrueferCodeEntry> + '_ {
        self.code()
            .iter()
            .enumerate()
            .map(|(i, p)| PrueferCodeEntry { i, parent: *p })
    }
}

impl<const N: usize> az_discrete_opt::log::ShortForm for PrueferCode<N> {
    fn short_form(&self) -> String {
        Tree::from(self).short_form()
    }
}

impl<const N: usize> From<&PrueferCode<N>> for Tree<N> {
    fn from(value: &PrueferCode<N>) -> Self {
        Tree::from_pruefer_code(value.code())
    }
}

#[derive(Debug)]
pub struct Tree<const N: usize> {
    pub parent: [Option<usize>; N],
}

impl<const N: usize> Tree<N> {
    pub fn generate(rng: &mut impl rand::Rng) -> Self {
        let mut code: [usize; N] = [0; N];
        (0..(N - 2)).for_each(|i| code[i] = rng.gen_range(0..N));
        Self::from_pruefer_code(&code[..N - 2])
    }

    fn from_pruefer_code(code: &[usize]) -> Self {
        debug_assert_eq!(code.len(), N - 2);
        debug_assert!(code.iter().all(|i| i.lt(&N)));
        // todo! there's a smarter algorithm here involving min-heaps
        let mut degrees = [1; N];
        for i in code {
            degrees[*i] += 1;
        }
        let mut parent = [None; N];
        for a in code {
            for b in 0..N {
                if degrees[b] == 1 {
                    // dbg!(a, b, parent, degrees);
                    parent[b] = Some(*a);
                    degrees[b] -= 1;
                    degrees[*a] -= 1;
                    break;
                }
            }
        }
        // find the last 2 vertices with nonzero degree
        let mut nonzero_degrees = degrees.iter().enumerate().filter(|(_, &d)| d.ne(&0));
        let (a, _) = nonzero_degrees.next().unwrap();
        let (b, _) = nonzero_degrees.next().unwrap();
        debug_assert!(nonzero_degrees.next().is_none());
        parent[a] = Some(b);
        parent[b] = None;
        assert!(
            parent.iter().filter(|p| p.is_none()).count() == 1,
            "code = {:?}, parent = {:?}",
            code,
            parent,
        );
        Self { parent }
    }

    pub fn edges(&self) -> impl Iterator<Item = Edge> + '_ {
        self.parent
            .iter()
            .enumerate()
            .filter_map(|(i, &p)| p.map(|p| Edge::new(i, p)))
    }

    pub fn conjecture_2_1_cost(&self) -> Conjecture2Dot1Cost {
        let a = self.adjacency_matrix();
        let eigs: Vec<faer::complex_native::c64> = a.eigenvalues();
        let lambda_1 = eigs
            .into_iter()
            .max_by(|a, b| a.re.partial_cmp(&b.re).unwrap())
            .unwrap()
            .re;
        assert!(lambda_1 > 1.4, "{a:?}",);
        let matching = self.maximum_matching();
        Conjecture2Dot1Cost { matching, lambda_1 }
    }

    pub fn adjacency_matrix(&self) -> faer::Mat<f64> {
        let mut a = faer::Mat::zeros(N, N);
        const ZERO: f64 = 0.0001;
        for i in 0..N {
            a[(i, i)] = ZERO;
        }
        for edge in self.edges() {
            let (v, u) = edge.vertices();
            a[(v, u)] = 1.0;
            a[(u, v)] = 1.0;
        }
        a
    }

    pub fn maximum_matching(&self) -> Vec<Edge> {
        // todo! basic algorithm
        #[derive(Clone, Copy)]
        enum VertexState {
            Unseen,
            Enqueued,
            Removed,
        }
        let mut states: [VertexState; N] = core::array::from_fn(|_| VertexState::Enqueued);
        for parent in self.parent.iter().flatten() {
            states[*parent] = VertexState::Unseen;
        }
        let mut queue = states
            .iter()
            .enumerate()
            .filter_map(|(i, b)| match b {
                VertexState::Enqueued => Some(i),
                _ => None,
            })
            .collect::<VecDeque<_>>();
        let mut matching = Vec::new();
        // before removing a vertex, enqueue its parent if it is not already seen
        while let Some(v) = queue.pop_back() {
            // if v is a root, skip it
            let p_v = match self.parent[v] {
                Some(p_v) => p_v,
                None => {
                    states[v] = VertexState::Removed;
                    continue;
                }
            };
            // v is seen, but if v is removed, make sure its parent is enqueued
            match states[v] {
                VertexState::Unseen => unreachable!(),
                VertexState::Enqueued => {}
                VertexState::Removed => {
                    if let VertexState::Unseen = states[p_v] {
                        states[p_v] = VertexState::Enqueued;
                        queue.push_back(p_v);
                        continue;
                    }
                }
            }
            states[v] = VertexState::Removed;
            // we removed v, but can it be matched?
            match states[p_v] {
                VertexState::Unseen => {
                    states[p_v] = VertexState::Removed;
                    matching.push(Edge::new(v, p_v));
                    queue.push_back(p_v);
                }
                VertexState::Enqueued => {
                    states[p_v] = VertexState::Removed;
                    matching.push(Edge::new(v, p_v));
                }
                VertexState::Removed => {}
            }
            // todo!()
        }
        // todo!();
        matching
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn wikipedia_prufer_code_has_correct_edges() {
        // https://en.wikipedia.org/wiki/Pr%C3%BCfer_sequence
        let code = [3, 3, 3, 4];
        let tree = Tree::<6>::from_pruefer_code(&code);
        let mut edges = tree.edges().collect::<Vec<_>>();
        edges.sort_by_key(|e| e.max());
        let expected_edges = vec![
            Edge::new(0, 3),
            Edge::new(1, 3),
            Edge::new(2, 3),
            Edge::new(3, 4),
            Edge::new(4, 5),
        ];
        assert_eq!(edges, expected_edges);
    }

    #[test]
    fn this_double_star_graph_has_matching_number_two_as_a_tree() {
        let tree_edges = [
            [3, 1],
            [12, 1],
            [16, 1],
            [18, 1],
            [19, 0],
            [19, 1],
            [19, 2],
            [19, 4],
            [19, 5],
            [19, 6],
            [19, 7],
            [19, 8],
            [19, 9],
            [19, 10],
            [19, 11],
            [19, 13],
            [19, 14],
            [19, 15],
            [19, 17],
        ];
        let mut edges: [Option<usize>; 20] = core::array::from_fn(|_| None);
        for [v, u] in tree_edges.iter() {
            edges[*v] = Some(*u);
        }
        let tree = Tree::<20> { parent: edges };
        let matching = tree.maximum_matching();
        assert_eq!(matching.len(), 2);

        let cost = tree.conjecture_2_1_cost();
        assert_eq!(cost.matching.len(), 2);
    }
}
