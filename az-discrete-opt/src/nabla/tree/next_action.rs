use petgraph::visit::EdgeRef;

use super::{EdgeIndex, NodeIndex, SearchTree};

pub(crate) enum NextAction {
    Visited(EdgeIndex),
    Unvisited(usize),
}

impl<P> SearchTree<P> {
    pub(crate) fn next_action(&self, state_pos: NodeIndex, n_as_tol: u32) -> Option<NextAction> {
        if !self.tree[state_pos].is_active() {
            return None;
        }
        let revisit_choice = self.revisit_choice(state_pos);
        if let Some((e, n_t_as, _)) = revisit_choice {
            if n_t_as < n_as_tol {
                return Some(NextAction::Visited(e));
            }
        }
        let max_curiosity = self.max_curiosity(state_pos);
        match max_curiosity {
            Some(max_curiosity) => Some(NextAction::Unvisited(max_curiosity)),
            None => revisit_choice.map(|(i, _, _)| NextAction::Visited(i)),
        }
    }

    pub(crate) fn revisit_choice(
        &self,
        state_pos: NodeIndex,
    ) -> Option<(EdgeIndex, u32, f32)> {
        self.tree
            .edges_directed(state_pos, petgraph::Direction::Outgoing)
            .filter_map(|e| {
                let child_weight = &self.tree[e.target()];
                if child_weight.is_active() {
                    Some((e.id(), child_weight.n_t(), child_weight.c_t_star))
                } else {
                    None
                }
                // child_weight
                //     .n_t
                //     .try_active()
                //     .map(|n_t| (e.id(), n_t, child_weight.c_t_star))
                // child_weight.n_t.map(|n_as| {
                //     (e.id(), n_as, child_weight.c_t_star)
                // })
                // todo!();
            })
            .min_by(|(_, n1, c_star1), (_, n2, c_star2)| {
                n1.cmp(n2).then(c_star1.partial_cmp(c_star2).unwrap())
            })
    }

    pub(crate) fn max_curiosity(&self, state_pos: NodeIndex) -> Option<usize> {
        let c_s: f32 = self.tree[state_pos].c;
        let c_t_star_values = self
            .tree
            .neighbors_directed(state_pos, petgraph::Direction::Outgoing)
            .map(|n_as| self.tree[n_as].c_t_star)
            .collect::<Box<_>>();
        let range = &self.tree.node_weight(state_pos).unwrap().actions;
        let range = (range.start as usize)..(range.end as usize);
        let predictions = &self.predictions[range.clone()];
        let c_theta_star_values = predictions
            .iter()
            .enumerate()
            .filter_map(|(i, prediction)| match prediction.edge_id {
                Some(_) => None,
                None => Some((i, c_s - prediction.g_theta_sa)),
            });
        if c_t_star_values.is_empty() {
            c_theta_star_values
                .min_by(|(_, g1), (_, g2)| g1.partial_cmp(g2).unwrap())
                .map(|(i, _)| i + range.start)
        } else {
            c_theta_star_values
                .map(|(i, c_theta_star)| {
                    let curiosity: f32 = c_t_star_values
                        .iter()
                        .map(|c_t_star| (c_t_star - c_theta_star).abs().sqrt())
                        .sum();
                    (i, curiosity)
                })
                .max_by(|(_, c1), (_, c2)| c1.partial_cmp(c2).unwrap())
                .map(|(i, _)| i + range.start)
        }
    }
}
