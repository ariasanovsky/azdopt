use crate::{
    bitset::{primitive::B32, Bitset},
    simple_graph::edge::Edge,
};

use super::ConnectedBitsetGraph;

impl<const N: usize, B> ConnectedBitsetGraph<N, B> {
    pub fn fast_cut_edges(&self) -> impl core::iter::Iterator<Item = Edge> + '_
    where
        B: Bitset + Clone + core::fmt::Debug + core::fmt::Display + PartialEq,
        B::Bits: Clone,
    {
        debug_assert!(N < 32, "N >= 32 not yet supported");
        #[derive(Debug)]
        struct Block<B> {
            block_positions_below: B, // nonempty
            elements: B,              // nonempty
            host: usize,
            kind: BlockKind,
        }
        #[derive(Debug)]
        enum BlockKind {
            Root,
            Cut { neighbor: usize },
        }
        let mut blocks: Vec<Block<B>> = vec![Block {
            block_positions_below: unsafe { B::singleton_unchecked(0) },
            elements: unsafe { B::singleton_unchecked(0) },
            host: 0,
            kind: BlockKind::Root,
        }];
        let mut active_block_indices = unsafe { B::singleton_unchecked(0) };
        let mut inserted_into: [usize; N] = [0; N];
        let mut explored_vertices = unsafe { B::singleton_unchecked(0) };
        let mut new_vertices = self.neighborhoods[0].clone();
        dbg!(
            &new_vertices,
            &explored_vertices,
            &inserted_into,
            &active_block_indices
        );
        while let Some(v) = new_vertices.remove_max() {
            dbg!(
                v,
                &new_vertices,
                &explored_vertices,
                &inserted_into,
                &active_block_indices
            );
            debug_assert!(!explored_vertices.contains(v).unwrap());
            let neighbors = &self.neighborhoods[v as usize];
            let unexplored_neighbors = neighbors.minus(&explored_vertices);
            new_vertices.union_assign(&unexplored_neighbors);
            let explored_neighbors = neighbors.intersection(&explored_vertices);

            debug_assert!(!explored_neighbors.is_empty());
            if explored_neighbors.is_singleton() {
                let u = unsafe { explored_neighbors.max_unchecked() } as usize;
                let u_i = inserted_into[u];
                let mut block_positions_below = blocks[u_i].block_positions_below.clone();
                let v_i = blocks.len() as u32;
                debug_assert!(!block_positions_below.contains(v_i).unwrap());
                unsafe { block_positions_below.add_or_remove_unchecked(v_i) };
                debug_assert!(!active_block_indices.contains(v_i).unwrap());
                unsafe { active_block_indices.add_or_remove_unchecked(v_i) };

                let block = Block {
                    block_positions_below,
                    elements: unsafe { B::singleton_unchecked(v) },
                    host: v as _,
                    kind: BlockKind::Cut { neighbor: u as _ },
                };
                dbg!(&block);
                blocks.push(block);
                inserted_into[v as usize] = v_i as usize;
            } else {
                let mut h_union = B::empty();
                let mut h_intersection = active_block_indices.clone();
                for u in explored_neighbors.iter() {
                    let u_i = inserted_into[u];
                    let blocks_below_u_i = &blocks[u_i].block_positions_below;
                    h_union.union_assign(blocks_below_u_i);
                    h_intersection.intersection_assign(blocks_below_u_i);
                }
                h_union.intersection_assign(&active_block_indices);
                debug_assert!(!h_intersection.is_empty());
                let h_i = unsafe { h_intersection.max_unchecked() };
                let deactivated_blocks = h_union.minus(&h_intersection);
                println!("h_union = {h_union}");
                println!("h_intersection = {h_intersection}");
                println!("deactivated_blocks = {deactivated_blocks}");
                if deactivated_blocks.is_empty() {
                    let b_i = &mut blocks[h_i as usize].elements;
                    debug_assert!(!b_i.contains(v).unwrap());
                    unsafe { b_i.add_or_remove_unchecked(v) };
                    inserted_into[v as usize] = h_i as usize;
                } else {
                    // dbg!(&blocks, &inserted_into);
                    println!("h_union = {h_union}");
                    println!("h_intersection = {h_intersection}");
                    println!("deactivated_blocks = {deactivated_blocks}");
                    println!("merge into b_{{{h_i}}}");
                    for i in deactivated_blocks.iter() {
                        debug_assert_ne!(i, h_i as usize);
                        let b_i = blocks[i].elements.clone();
                        blocks[h_i as usize].elements.union_assign(&b_i);
                    }
                    unsafe { blocks[h_i as usize].elements.add_or_remove_unchecked(v) };
                    println!("now blocks[{h_i}] = {}", &blocks[h_i as usize].elements);
                    debug_assert_eq!(
                        active_block_indices.intersection(&deactivated_blocks),
                        deactivated_blocks
                    );
                    active_block_indices.symmetric_difference_assign(&deactivated_blocks);
                    println!("active blocks: {active_block_indices}");
                    inserted_into[v as usize] = h_i as usize;
                }
            }
            debug_assert!(!explored_neighbors.contains(v).unwrap());
            unsafe { explored_vertices.add_or_remove_unchecked(v) };
        }

        let cuts = active_block_indices
            .iter()
            .filter_map(|i| {
                let block = &blocks[i];
                let Block {
                    block_positions_below: _,
                    elements: _,
                    host,
                    kind,
                } = block;
                match kind {
                    BlockKind::Root => None,
                    BlockKind::Cut { neighbor } => Some(Edge::new(*host, *neighbor)),
                }
            })
            .collect::<Vec<_>>();
        cuts.into_iter()
    }
}

#[cfg(test)]
mod tests {
    // use std::mem::MaybeUninit;

    use crate::{simple_graph::{connected_bitset_graph::ConnectedBitsetGraph, edge::Edge}, bitset::primitive::B32};
    type G<const N: usize> = ConnectedBitsetGraph<N, B32>;

    const S_4: [[usize; 4]; 24] = [
        [0, 1, 2, 3],
        [0, 1, 3, 2],
        [0, 2, 1, 3],
        [0, 2, 3, 1],
        [0, 3, 1, 2],
        [0, 3, 2, 1],
        [1, 0, 2, 3],
        [1, 0, 3, 2],
        [1, 2, 0, 3],
        [1, 2, 3, 0],
        [1, 3, 0, 2],
        [1, 3, 2, 0],
        [2, 0, 1, 3],
        [2, 0, 3, 1],
        [2, 1, 0, 3],
        [2, 1, 3, 0],
        [2, 3, 0, 1],
        [2, 3, 1, 0],
        [3, 0, 1, 2],
        [3, 0, 2, 1],
        [3, 1, 0, 2],
        [3, 1, 2, 0],
        [3, 2, 0, 1],
        [3, 2, 1, 0],
    ];

    // #[derive(Debug, PartialEq, Eq)]
    // struct PermutationAndInverse<const N: usize> {
    //     permutation: [usize; N],
    //     inverse: [usize; N],
    // }

    // impl<const N: usize> PermutationAndInverse<N> {
    //     fn new(f: &[usize; N]) -> Result<Self, ()> {
    //         let mut f_range_set: [bool; N] = [false; N];
    //         let mut forward: [MaybeUninit<usize>; N] = MaybeUninit::uninit_array();
    //         let mut backward: [MaybeUninit<usize>; N] = MaybeUninit::uninit_array();
    //         for (i, &f_i) in f.into_iter().enumerate() {
    //             if f_range_set[f_i] {
    //                 return Err(())
    //             }
    //             f_range_set[f_i] = true;
    //             forward[i].write(f_i);
    //             backward[f_i].write(i);
    //         }
    //         let forward = unsafe { MaybeUninit::array_assume_init(forward) };
    //         let backward = unsafe { MaybeUninit::array_assume_init(backward) };
    //         Ok(Self {
    //             permutation: forward,
    //             inverse: backward,
    //         })
    //     }

    //     fn apply(&self, i: usize) -> usize {
    //         self.permutation[i]
    //     }

    //     fn apply_inverse(&self, i: usize) -> usize {
    //         self.inverse[i]
    //     }
    // }

    // #[test]
    // fn constant_zero_function_is_not_a_permutation() {
    //     let f: [usize; 4] = [0; 4];
    //     assert!(PermutationAndInverse::new(&f).is_err())
    // }

    // #[test]
    // fn identity_function_is_a_permutation() {
    //     let f: [usize; 4] = [0, 1, 2, 3];
    //     assert_eq!(
    //         PermutationAndInverse::new(&f).unwrap(),
    //         PermutationAndInverse {
    //             permutation: [0, 1, 2, 3],
    //             inverse: [0, 1, 2, 3],
    //         }
    //     )
    // }

    fn verify_cut_edges_after_permutation<const N: usize>(
        edges: &[(usize, usize)],
        cut_edges: &[(usize, usize)],
        sigma: &[usize],
    ) {
        let edges = edges
            .into_iter()
            .map(|(u, v)| (sigma[*u], sigma[*v]))
            .collect::<Vec<_>>();
        let graph: ConnectedBitsetGraph<N, B32> = edges[..].try_into().unwrap();
        let cut_edges = cut_edges
            .into_iter()
            .map(|(u, v)| Edge::new(sigma[*u], sigma[*v]))
            .collect::<Vec<_>>();
        assert_eq!(&graph.cut_edges().collect::<Vec<_>>(), &cut_edges)
    }

    fn old_and_new_cut_edge_methods_are_identical<const N: usize>(graph: &ConnectedBitsetGraph<N, B32>) {
        let mut cut_edges = graph.cut_edges().collect::<Vec<_>>();
        let mut new_cut_edges = graph.fast_cut_edges().collect::<Vec<_>>();
        cut_edges.sort_by(|a, b| match a.max.cmp(&b.max) {
            std::cmp::Ordering::Equal => a.min.cmp(&b.min),
            s => s,
        });
        new_cut_edges.sort_by(|a, b| match a.max.cmp(&b.max) {
            std::cmp::Ordering::Equal => a.min.cmp(&b.min),
            s => s,
        });
        assert_eq!(
            cut_edges, new_cut_edges,
            "graph:\n{graph}\ncut_edges: {cut_edges:?}\nnew_cut_edges: {new_cut_edges:?}",
        )
    }

    #[test]
    fn old_and_new_cut_edge_methods_are_identical_for_random_graphs_on_ten_vertices() {
        let mut rng = rand::thread_rng();
        for _ in 0..1_000 {
            let g = G::<30>::generate(0.05, &mut rng);
            old_and_new_cut_edge_methods_are_identical(&g);
        }
    }

    #[test]
    fn old_and_new_cut_edge_methods_are_identical_for_c4() {
        let graph: G<4> = [(0, 1), (1, 2), (2, 3), (3, 0)]
            .as_ref()
            .try_into()
            .unwrap();
        old_and_new_cut_edge_methods_are_identical(&graph);
    }

    #[test]
    fn old_and_new_cut_edge_methods_are_identical_for_paw() {
        let graph: G<4> = [(0, 1), (1, 2), (1, 3), (2, 3)]
            .as_ref()
            .try_into()
            .unwrap();
        old_and_new_cut_edge_methods_are_identical(&graph);
    }

    #[test]
    fn old_and_new_cut_edge_methods_are_identical_for_house() {
        let graph: G<5> = [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (3, 4)]
            .as_ref()
            .try_into()
            .unwrap();
        old_and_new_cut_edge_methods_are_identical(&graph);
    }

    #[test]
    fn cycle_on_four_vertices_has_no_cut_edges() {
        let edges = [(0, 1), (1, 2), (2, 3), (3, 0)];
        let cut_edges = [];
        for sigma in S_4 {
            verify_cut_edges_after_permutation::<4>(&edges, &cut_edges, &sigma)
        }
    }

    #[test]
    fn paw_graph_has_one_cut_edge() {
        let edges = [(0, 1), (1, 2), (1, 3), (2, 3)].as_ref();
        let cut_edges = [(0, 1)];
        for sigma in S_4 {
            verify_cut_edges_after_permutation::<4>(&edges, &cut_edges, &sigma)
        }
    }
}
