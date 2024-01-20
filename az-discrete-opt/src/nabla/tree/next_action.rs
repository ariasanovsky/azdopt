use petgraph::visit::EdgeRef;

use crate::nabla::space::NablaStateActionSpace;

use super::{SearchTree, EdgeIndex, NodeIndex};

const CONCERN_TOLERANCE: f32 = 0.1;

pub(crate) enum NextAction {
    Visited(EdgeIndex),
    Unvisited(usize),
}

impl<P> SearchTree<P> {
    pub(crate) fn next_action<Space: NablaStateActionSpace>(&self, space: &Space, state_pos: NodeIndex) -> Option<NextAction> {
        let max_concern = self.max_concern(space, state_pos);
        match max_concern {
            Some((e, concern)) => {
                if concern > CONCERN_TOLERANCE {
                    return Some(NextAction::Visited(e))
                }
            },
            _ => {},
        }
        let max_curiosity = self.max_curiosity(space, state_pos);
        match max_curiosity {
            Some(max_curiosity) => Some(NextAction::Unvisited(max_curiosity)),
            None => max_concern.map(|(i, _)| NextAction::Visited(i)),
        }
    }

    pub(crate) fn max_concern<Space: NablaStateActionSpace>(&self, space: &Space, state_pos: NodeIndex) -> Option<(EdgeIndex, f32)> {
        let c_s = self.tree[state_pos].c;
        self.tree
            .edges_directed(state_pos, petgraph::Direction::Outgoing)
            .filter_map(|e| {
                let child_weight = self.tree.node_weight(e.target()).unwrap();
                child_weight.n_t.map(|n_as| {
                    let arc_weight = self.tree.edge_weight(e.id()).unwrap();
                    let c_as = child_weight.c;
                    let c_as_star = child_weight.c_t_star;
                    let h_t_sa = space.h_sa(c_s, c_as, c_as_star);
                    let prediction = &self.predictions[arc_weight.prediction_pos];
                    let h_theta_sa = prediction.h_theta_sa;
                    let concern = (h_t_sa - h_theta_sa).powi(2) / n_as.get() as f32;
                    (e.id(), concern)
                })
            })
            .max_by(|(_, c1), (_, c2)| {
                c1.partial_cmp(c2).unwrap()
            })
    }

    pub(crate) fn max_curiosity<Space: NablaStateActionSpace>(&self, space: &Space, state_pos: NodeIndex) -> Option<usize> {
        let c_s: f32 = self.tree[state_pos].c;
        let g_values =
            self.tree.edges_directed(state_pos, petgraph::Direction::Outgoing)
            .map(|e| {
                let c_as_star: f32 = self.tree[e.target()].c_t_star;
                let g_t_sa = c_s - c_as_star;
                g_t_sa
                // e.weight().g_t_sa
            })
            .collect::<Box<_>>();
        let range = &self.tree.node_weight(state_pos).unwrap().actions;
        let predictions = &self.predictions[range.clone()];
        let foo = predictions.iter().enumerate().filter_map(|(i, prediction)| {
            match prediction.edge_id {
                Some(_) => None,
                None => Some({
                    let g_theta_sa = prediction.g_theta_sa;
                    let curiosity: f32 = g_values.iter().map(|g_t_sa| {
                        (g_t_sa - g_theta_sa).abs().sqrt()
                    }).sum();
                    (i, curiosity)
                }),
            }
        });
        foo.max_by(|(_, c1), (_, c2)| {
            c1.partial_cmp(c2).unwrap()
        }).map(|(i, _)| i + range.start)
    }
}
