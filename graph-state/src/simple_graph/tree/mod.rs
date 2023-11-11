use super::edge::Edge;

pub mod sparse6;
mod display;
mod state;

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
}

impl<const N: usize> From<PrueferCode<N>> for Tree<N> {
    fn from(value: PrueferCode<N>) -> Self {
        Tree::from_pruefer_code(value.code())
    }
}

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
                    dbg!(a, b, parent, degrees);
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
        Self { parent }
    }

    pub fn edges(&self) -> impl Iterator<Item = Edge> + '_ {
        self.parent
            .iter()
            .enumerate()
            .filter_map(|(i, &p)| p.map(|p| Edge::new(i, p)))
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
}