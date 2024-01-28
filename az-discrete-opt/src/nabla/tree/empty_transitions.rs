use std::collections::BTreeMap;

use crate::nabla::tree::NodeIndex;

use super::{EdgeIndex, SearchTree};

struct CascadeInfo<I> {
    current_nodes: BTreeMap<NodeIndex, I>,
    next_nodes: BTreeMap<NodeIndex, I>,
}

impl<I> CascadeInfo<I> {
    fn new(state_pos: NodeIndex, child_info: I) -> Self {
        Self {
            current_nodes: core::iter::once((state_pos, child_info)).collect(),
            next_nodes: BTreeMap::new(),
        }
    }

    fn pop_front(&mut self) -> Option<(NodeIndex, I)> {
        self.current_nodes.pop_first().or_else(|| {
            core::mem::swap(&mut self.current_nodes, &mut self.next_nodes);
            self.current_nodes.pop_first()
        })
    }

    fn upsert(
        &mut self,
        state_pos: NodeIndex,
        new_child_info: I,
        update: impl FnOnce(&mut I, &I),
    ) -> &I {
        let entry = self.next_nodes.entry(state_pos);
        let value = entry
            .and_modify(|child_info| {
                update(child_info, &new_child_info);
            })
            .or_insert(new_child_info);
        value
    }
}

#[derive(Clone, Copy)]
struct Info {
    c_t_star: f32,
    newly_exhausted_children: u32,
}

impl<P> SearchTree<P> {
    pub(crate) fn cascade_new_terminal(&mut self, edge_id: EdgeIndex) {
        let a_t = &self.tree.raw_edges()[edge_id.index()];
        let s_t = &self.tree[a_t.target()];
        debug_assert_eq!(s_t.n_t(), 0);
        debug_assert!(!s_t.is_active());
        let c_t = s_t.c_t_star;
        let parent_index = a_t.source();
        let info = Info {
            c_t_star: c_t,
            newly_exhausted_children: 1,
        };
        let mut cascade_info = CascadeInfo::new(parent_index, info);
        while let Some((child_index, ancestor_info)) = cascade_info.pop_front() {
            let Info { c_t_star, newly_exhausted_children: exhausted_children } = ancestor_info;
            let child = &mut self.tree[child_index];
            child.exhausted_children += exhausted_children;
            if child.c_t_star > c_t_star {
                child.c_t_star = c_t_star;
            } else {
                child.n_t += 1;
            }
            let new_child_info = Info {
                c_t_star,
                newly_exhausted_children: if child.is_active() { 0 } else { 1 },
            };
            let mut neigh = self
                .tree
                .neighbors_directed(child_index, petgraph::Direction::Incoming)
                .detach();
            while let Some(parent_id) = neigh.next_node(&self.tree) {
                let update = |child_info: &mut Info, new_child_info: &Info| {
                    child_info.c_t_star = child_info.c_t_star.min(new_child_info.c_t_star);
                    child_info.newly_exhausted_children += new_child_info.newly_exhausted_children;
                };
                cascade_info.upsert(parent_id, new_child_info, update);
            }
        }
    }

    pub(crate) fn cascade_old_node(&mut self, edge_id: EdgeIndex) {
        let a_t = &self.tree.raw_edges()[edge_id.index()];
        let s_t = &self.tree[a_t.target()];
        let n_t_s_t = s_t.n_t();
        let newly_exhausted_children = if s_t.is_active() { 0 } else { 1 };
        let c_t = s_t.c_t_star;
        let parent_index = a_t.source();
        let info = Info {
            c_t_star: c_t,
            newly_exhausted_children,
        };
        let mut cascade_info = CascadeInfo::new(parent_index, info);
        while let Some((child_index, ancestor_info)) = cascade_info.pop_front() {
            let Info { c_t_star, newly_exhausted_children: exhausted_children } = ancestor_info;
            let child = &mut self.tree[child_index];
            child.exhausted_children += exhausted_children;
            if child.c_t_star > c_t_star {
                child.c_t_star = c_t_star;
            } else {
                child.n_t += 1;
            }
            child.n_t = child.n_t.max(n_t_s_t);
            let new_child_info = Info {
                c_t_star,
                newly_exhausted_children: if child.is_active() { 0 } else { 1 },
            };
            let mut neigh = self
                .tree
                .neighbors_directed(child_index, petgraph::Direction::Incoming)
                .detach();
            while let Some(parent_id) = neigh.next_node(&self.tree) {
                let update = |child_info: &mut Info, new_child_info: &Info| {
                    child_info.c_t_star = child_info.c_t_star.min(new_child_info.c_t_star);
                    child_info.newly_exhausted_children += new_child_info.newly_exhausted_children;
                };
                cascade_info.upsert(parent_id, new_child_info, update);
            }
        }
    }
}
