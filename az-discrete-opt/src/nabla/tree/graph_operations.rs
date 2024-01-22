use petgraph::stable_graph::{NodeIndex, EdgeIndex};

use crate::nabla::{space::NablaStateActionSpace, tree::arc_weight::ActionPrediction};

use super::{SearchTree, state_weight::{StateWeight, NumLeafDescendants}, arc_weight::ActionWeight};

impl<P> SearchTree<P> {
    pub(crate) fn add_node(&mut self, p: P, weight: StateWeight) -> NodeIndex
    where
        P: Ord,
    {
        let index = self.tree.add_node(weight);
        let old_index = self.positions.insert(p, index);
        debug_assert!(old_index.is_none());
        index
    }

    pub(crate) fn add_arc(&mut self, parent: NodeIndex, child: NodeIndex, prediction_pos: usize) -> EdgeIndex {
        let arc_weight = ActionWeight { prediction_pos };
        // print!("\tnew arc_weight: {arc_weight:?}");
        let arc_index = self.tree.add_edge(parent, child, arc_weight);
        self.predictions[prediction_pos].edge_id = Some(arc_index);
        // println!(" on {arc_index:?}");
        arc_index
    }

    pub(crate) fn add_actions<Space: NablaStateActionSpace>(
        &mut self,
        id: NodeIndex,
        space: &Space,
        state: &Space::State,
        h_theta: &[f32],
    ) {
        let node_weight = self.tree.node_weight_mut(id).unwrap();
        let c = node_weight.c;
        let start = self.predictions.len();
        let foo =
            space.action_data(state)
            .map(|(a_id, r_sa)| {
                let h_theta_sa = h_theta[a_id];
                let g_theta_sa = space.g_theta_star_sa(c, r_sa, h_theta_sa);
                ActionPrediction {
                    a_id,
                    // h_theta_sa,
                    g_theta_sa,
                    edge_id: None,
                }
            });
        self.predictions.extend(foo);
        let end = self.predictions.len();
        debug_assert_ne!(start, end);
        node_weight.actions = (start as u32)..(end as u32);
    }

    pub(crate) fn update_exhaustion(&mut self, id: NodeIndex) -> &mut StateWeight {
        // println!("tupdating exhaustion for {id:?}");
        let action_range = &self.tree[id].actions;
        let action_range = (action_range.start as usize)..(action_range.end as usize);
        let actions = &self.predictions[action_range.clone()];
        // println!("actions: {actions:?}");
        let active = actions.iter().any(|a| {
            match a.edge_id {
                Some(edge_id) => {
                    // println!("\tedge_id: {edge_id:?}");
                    let e = &self.tree.raw_edges()[edge_id.index()];
                    let child = e.target();
                    // println!("child weight: {:?}", self.tree[child]);
                    match self.tree[child].n_t.try_active() {
                        Some(_a) => {
                            // println!("\t\tactive");
                            true
                        },
                        None => {
                            // println!("\t\tinactive");
                            false
                        },
                    }
                },
                None => {
                    // println!("\tactive");
                    true
                },
            }
        });
        // println!("active: {active}");
        let s_t = &mut self.tree[id];
        let n_t = &mut s_t.n_t;
        if !active {
            debug_assert!(n_t.try_active().is_some());
            n_t.mark_exhausted();
        }
        s_t
    }
}
