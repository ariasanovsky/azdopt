use core::marker::PhantomData;
use std::{array::from_fn, collections::VecDeque};

use super::{my_bitsets_to_refactor_later::B32, graph::{Neighborhoods, Tree}};

pub(crate) enum PartiallyExplored {}

#[derive(Clone)]
pub(crate) enum Insertion {
    Root,
    NewHost { previous_host: usize },
    Hosted { host: usize },
}

#[derive(Clone)]
pub struct BlockForest<const N: usize, State = ()> {
    pub(crate) hosted_vertices: [B32; N],
    pub(crate) hosts_below: [B32; N],
    pub(crate) insertion: [Option<Insertion>; N],
    pub(crate) active_hosts: B32,
    pub(crate) state: PhantomData<State>,
}

impl<const N: usize, S> BlockForest<N, S> {
    pub(crate) fn host(&self, u: usize) -> usize {
        self.insertion[u].as_ref().map(|i| match i {
            Insertion::Root => u,
            Insertion::NewHost { previous_host: _ } => u,
            Insertion::Hosted { host } => *host,
        }).unwrap_or(u)
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
        let Neighborhoods { neighborhoods } = neighborhoods;
        let mut seen_vertices = explored_vertices.clone();
        let mut new_neighborhoods = VecDeque::from([(root, neighborhoods[root].clone())]);
        while let Some((u, n_u)) = new_neighborhoods.pop_front() {
            // elements of `n_u` with a neighbor with in `n_u`
            let mut t_u = B32::empty();
            // explored vertices whose blocks merge with `u`'s
            let mut x_u = B32::empty();
            // elements of `n_u` which witness such a merge
            let mut a_u = B32::empty();
            n_u.iter().for_each(|v| {
                let n_v = &neighborhoods[v];
                let new_v = n_v.minus(&seen_vertices);
                if !new_v.is_empty() {
                    seen_vertices.union_assign(&new_v);
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
            let mut host_u = self.host(u);
            // merge blocks first
            if !x_u.is_singleton() {
                x_u.add_or_remove_unchecked(u);
                let mut hosts_of_blocks_to_merge = B32::empty();
                hosts_of_blocks_to_merge.insert_unchecked(host_u);
                x_u.iter().for_each(|v| {
                    let host_v = self.host(v);
                    hosts_of_blocks_to_merge.insert_unchecked(host_v);
                });
                if !hosts_of_blocks_to_merge.is_singleton() {
                    let mut meet_u = self.hosts_below[host_u].clone();
                    let mut join_u = self.hosts_below[host_u].clone();
                    hosts_of_blocks_to_merge.add_or_remove_unchecked(host_u);
                    hosts_of_blocks_to_merge.iter().for_each(|h| {
                        meet_u.intersection_assign(&self.hosts_below[h]);
                        join_u.union_assign(&self.hosts_below[h]);
                    });
                    meet_u.intersection_assign(&self.active_hosts);
                    let meet_host = meet_u.max_unchecked();
                    let mut meet_hosted_vertices = self.hosted_vertices[meet_host].clone();
                    join_u.intersection_assign(&self.active_hosts);
                    let hosts_to_deactivate = join_u.clone().symmetric_difference(&meet_u);
                    hosts_to_deactivate.iter().for_each(|h| {
                        meet_hosted_vertices.union_assign(&self.hosted_vertices[h]);
                    });
                    self.hosted_vertices[meet_host] = meet_hosted_vertices;
                    self.active_hosts.minus_assign(&hosts_to_deactivate);
                    host_u = meet_host;
                }
            }
            let mut new_block_vertices_u = t_u;
            new_block_vertices_u.union_assign(&a_u);
            self.hosted_vertices[host_u].union_assign(&new_block_vertices_u);
            new_block_vertices_u.iter().for_each(|v| {
                self.insertion[v] = Some(Insertion::Hosted { host: host_u });
                self.hosts_below[v] = self.hosts_below[host_u].clone();
                self.hosts_below[v].insert_unchecked(v);
            });
            // other vertices get added to the tree via a cut edge
            let c_u = n_u.symmetric_difference(&new_block_vertices_u);
            c_u.iter().for_each(|v| {
                self.insertion[v] = Some(Insertion::NewHost { previous_host: host_u });
            });
            self.active_hosts.union_assign(&c_u);
            explored_vertices.union_assign(&n_u);
        }
        todo!()
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
