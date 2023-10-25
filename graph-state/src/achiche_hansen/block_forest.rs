use core::marker::PhantomData;
use std::{array::from_fn, collections::VecDeque};

use super::{my_bitsets_to_refactor_later::B32, graph::{Neighborhoods, Tree}};

#[derive(Clone, Debug)]
pub(crate) enum PartiallyExplored {}

#[derive(Clone, Debug)]
pub(crate) enum Insertion {
    Root,
    NewHost { previous_host: usize },
    Hosted { host: usize },
}

impl Insertion {
    pub(crate) fn host(&self) -> Option<usize> {
        match self {
            Self::Root | Self::NewHost { previous_host: _ } => None,
            Self::Hosted { host } => Some(*host),
        }
    }
}

#[derive(Clone)]
pub struct BlockForest<const N: usize, State = PartiallyExplored> {
    pub(crate) hosted_vertices: [B32; N],
    pub(crate) hosts_below: [B32; N],
    pub(crate) insertion: [Option<Insertion>; N],
    pub(crate) active_hosts: B32,
    pub(crate) state: PhantomData<State>,
}

impl<const N: usize, S> BlockForest<N, S> {
    pub(crate) fn blocks(&self) -> Vec<B32> {
        self.active_hosts.iter().map(|h| self.hosted_vertices[h].clone()).collect()
    }
}

impl<const N: usize> BlockForest<N, PartiallyExplored> {
    pub fn new() -> Self {
        Self {
            hosted_vertices: from_fn(|_| B32::empty()),
            hosts_below: from_fn(|_| B32::empty()),
            insertion: from_fn(|_| None),
            active_hosts: B32::empty(),
            state: PhantomData,
        }
    }

    pub(crate) fn insert_root(&mut self, root: usize) {
        let Self {
            hosted_vertices,
            hosts_below,
            insertion,
            active_hosts,
            state: _,
        } = self;
        let hosted_vertices = &mut hosted_vertices[root];
        *hosted_vertices = B32::empty();
        hosted_vertices.insert_unchecked(root);
        active_hosts.insert_unchecked(root);
        let hosts_below = &mut hosts_below[root];
        *hosts_below = B32::empty();
        hosts_below.insert_unchecked(root);
        let insertion = &mut insertion[root];
        *insertion = Some(Insertion::Root);
    }

    pub(crate) fn explore_from(
        &mut self,
        neighborhoods: &Neighborhoods<N>,
        explored_vertices: &mut B32,
        root: usize,
    ) {
        let Self {
            hosted_vertices,
            hosts_below,
            insertion,
            active_hosts,
            state: _,
        } = self;
        let Neighborhoods { neighborhoods } = neighborhoods;
        println!("{:?}", neighborhoods.iter().map(|n| n.to_string()).collect::<Vec<_>>());
        explored_vertices.insert_unchecked(root);
        let mut seen_vertices = explored_vertices.clone();
        let n_root = neighborhoods[root].minus(&seen_vertices);
        if n_root.is_empty() {
            return;
        }
        seen_vertices.union_assign(&n_root);
        let mut new_neighborhoods = VecDeque::from([(root, n_root)]);
        while let Some((u, n_u)) = new_neighborhoods.pop_front() {
            println!("u = {u}\nn_{u} = {n_u}");
            println!("seen = {seen_vertices}\nexplored = {explored_vertices}");
            active_hosts.iter().for_each(|w| {
                println!("\tblock {w} = {}", hosted_vertices[w]);
            });
            // elements of `n_u` with a neighbor with in `n_u`
            let mut t_u = B32::empty();
            // explored vertices whose blocks merge with `u`'s
            let mut x_u = B32::empty();
            x_u.insert_unchecked(u);
            // elements of `n_u` which witness such a merge
            let mut a_u = B32::empty();
            n_u.iter().for_each(|v| {
                let n_v = &neighborhoods[v];
                let new_v = n_v.minus(&seen_vertices);
                if !new_v.is_empty() {
                    seen_vertices.union_assign(&new_v);
                    println!("new_{v} = {new_v}");
                    new_neighborhoods.push_back((v, new_v));
                }
                // the common neighbors of `u` and `v` belong in the block of `u`
                let n_uv = n_u.intersection(&n_v);
                t_u.union_assign(&n_uv);
                // the explored neighbors of `v`
                let x_v = n_v.intersection(&explored_vertices);
                // by definition, {u} is in `x_v`
                // if `x_v` contains more than `u`, `v` is in the block of `u` (we delay this insertion)
                // additionally, the corresponding blocks merge
                if !x_v.is_singleton() {
                    x_u.union_assign(&x_v);
                    a_u.insert_unchecked(v);
                }
            });
            println!("t_{u} = {t_u}\nx_{u} = {x_u}\na_{u} = {a_u}");
            let mut host_u = insertion[u].as_ref().map(Insertion::host).flatten().unwrap_or(u);
            // merge blocks first
            if !x_u.is_singleton() {
                x_u.add_or_remove_unchecked(u);
                let mut hosts_of_blocks_to_merge = B32::empty();
                hosts_of_blocks_to_merge.insert_unchecked(host_u);
                x_u.iter().for_each(|v| {
                    let host_v = insertion[v].as_ref().map(Insertion::host).flatten().unwrap_or(v);
                    hosts_of_blocks_to_merge.insert_unchecked(host_v);
                });
                if !hosts_of_blocks_to_merge.is_singleton() {
                    let mut meet_u = hosts_below[host_u].clone();
                    let mut join_u = hosts_below[host_u].clone();
                    hosts_of_blocks_to_merge.add_or_remove_unchecked(host_u);
                    hosts_of_blocks_to_merge.iter().for_each(|h| {
                        meet_u.intersection_assign(&hosts_below[h]);
                        join_u.union_assign(&hosts_below[h]);
                    });
                    meet_u.intersection_assign(&active_hosts);
                    let meet_host = meet_u.max_unchecked();
                    let mut meet_hosted_vertices = hosted_vertices[meet_host].clone();
                    join_u.intersection_assign(&active_hosts);
                    let hosts_to_deactivate = join_u.clone().symmetric_difference(&meet_u);
                    hosts_to_deactivate.iter().for_each(|h| {
                        meet_hosted_vertices.union_assign(&hosted_vertices[h]);
                    });
                    hosted_vertices[meet_host] = meet_hosted_vertices;
                    active_hosts.minus_assign(&hosts_to_deactivate);
                    host_u = meet_host;
                }
            }
            let mut new_block_vertices_u = t_u;
            new_block_vertices_u.union_assign(&a_u);
            hosted_vertices[host_u].union_assign(&new_block_vertices_u);
            new_block_vertices_u.iter().for_each(|v| {
                insertion[v] = Some(Insertion::Hosted { host: host_u });
            });
            // other vertices get added to the tree via a cut edge
            let c_u = n_u.symmetric_difference(&new_block_vertices_u);
            println!("c_{u} = {c_u}");
            c_u.iter().for_each(|v| {
                insertion[v] = Some(Insertion::NewHost { previous_host: host_u });
                hosts_below[v] = hosts_below[host_u].clone();
                hosts_below[v].insert_unchecked(v);
                hosted_vertices[v] = B32::empty();
                hosted_vertices[v].insert_unchecked(v);
            });
            active_hosts.union_assign(&c_u);
            // `u` is already explored
            // mark `n_u` as explored
            explored_vertices.union_assign(&n_u);
            
        }
    }

    pub(crate) fn assert_tree(self) -> BlockForest<N, Tree> {
        let Self {
            hosted_vertices: owned_vertices,
            hosts_below: owners_below,
            insertion: cut_edges,
            active_hosts: owners,
            state: _,
        } = self;
        BlockForest {
            hosted_vertices: owned_vertices,
            hosts_below: owners_below,
            insertion: cut_edges,
            active_hosts: owners,
            state: PhantomData,
        }
    }
}
