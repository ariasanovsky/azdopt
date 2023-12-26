use crate::bitset::Bitset;

use super::BitsetGraph;

impl<const N: usize, B> BitsetGraph<N, B> {
    pub fn to_graph6(&self) -> Vec<u8>
    where
        B: Bitset,
    {
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
