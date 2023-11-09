use az_discrete_opt::log::ShortForm;

use super::ConnectedBitsetGraph;

impl<const N: usize> ShortForm for ConnectedBitsetGraph<N> {
    fn short_form(&self) -> String {
        let graph6 = self.to_graph6();
        String::from_utf8(graph6).unwrap()
    }
}

impl<const N: usize> ConnectedBitsetGraph<N> {
    pub fn to_graph6(&self) -> Vec<u8> {
        let mut graph6: Vec<u8> = Vec::new();

        // Add the number of vertices to the graph6 string
        if N <= 62 {
            graph6.push(N as u8 + 63);
        } else {
            unimplemented!("graphs on more than 62 vertices")
            // graph6.push((126 as u8) as char);
            // graph6.push(((N >> 12) + 63) as char);
            // graph6.push((((n >> 6) & 63) + 63) as char);
            // graph6.push(((n & 63) + 63) as char);
        }

        // Add the adjacency matrix to the graph6 string
        let mut bool_edges = self.edge_bools().collect::<Vec<_>>();
        let padding = (6 - (bool_edges.len() % 6)) % 6;
        bool_edges.extend(vec![false; padding]);
        bool_edges.chunks_exact(6).for_each(|chunk| {
            let mut byte = 0;
            chunk.iter().rev().enumerate().for_each(|(i, &b)| {
                if b {
                    byte |= 1 << i;
                }
            });
            graph6.push(byte + 63);
        });
        graph6
    }
}

#[cfg(test)]
mod test {
    use crate::simple_graph::bitset_graph::BitsetGraph;

    #[test]
    fn complete_graph_on_two_vertices_has_correct_g6_string() {
        let graph: BitsetGraph<2> = [(0, 1)].as_ref().try_into().unwrap();
        let g6 = graph.to_graph6();
        debug_assert_eq!(&g6, &[b'A', b'_'])
    }

    #[test]
    fn the_two_regular_graphs_on_order_eight_from_the_g6_standard_have_correct_g6_strings() {
        // g6 standard: http://users.cecs.anu.edu.au/~bdm/data/formats.html
        let edges: [Vec<(usize, usize)>; 3] = [
            vec![(0, 4), (4, 1), (1, 5), (5, 0), (2, 6), (6, 3), (3, 7), (7, 2)],
            vec![(0, 4), (4, 1), (1, 6), (6, 3), (3, 7), (7, 2), (2, 5), (5, 0)],
            vec![(0, 3), (3, 5), (5, 0), (1, 4), (4, 7), (7, 2), (2, 6), (6, 1)],
        ];
        let graphs = edges.map(|edges| {
            let graph: BitsetGraph<8> = edges[..].try_into().unwrap();
            graph
        });
        let rhs = graphs.map(|graph| graph.to_graph6());
        let g6 = [b"G?r@`_", b"G?qa`_", b"GCQR@O"];
        debug_assert_eq!(&rhs, &g6)
    }
}
