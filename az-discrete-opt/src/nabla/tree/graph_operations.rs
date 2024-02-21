use petgraph::stable_graph::{EdgeIndex, NodeIndex};

use crate::nabla::{space::DfaWithCost, tree::arc_weight::ActionPrediction};

use super::{arc_weight::ActionWeight, state_weight::StateWeight, SearchTree};

impl SearchTree {
    pub(crate) fn add_node(&mut self, weight: StateWeight) -> NodeIndex
    {
        let index = self.tree.add_node(weight);
        // let old_index = self.positions.insert(p, index);
        // debug_assert!(old_index.is_none());
        index
    }

    pub(crate) fn add_arc(
        &mut self,
        parent: NodeIndex,
        child: NodeIndex,
        prediction_pos: usize,
    ) -> EdgeIndex {
        let arc_weight = ActionWeight { prediction_pos };
        // print!("\tnew arc_weight: {arc_weight:?}");
        let arc_index = self.tree.add_edge(parent, child, arc_weight);
        self.predictions[prediction_pos].edge_id = Some(arc_index);
        // println!(" on {arc_index:?}");
        arc_index
    }

    pub(crate) fn add_actions<Space: DfaWithCost>(
        &mut self,
        id: NodeIndex,
        space: &Space,
        state: &Space::State,
        h_theta: &[f32],
        sample: &SamplePattern,
    ) {
        let node_weight = self.tree.node_weight_mut(id).unwrap();
        let c = node_weight.c;
        let start = self.predictions.len();
        let predictions = space.action_data(state).map(|(a_id, r_sa)| {
            let h_theta_sa = h_theta[a_id];
            let g_theta_sa = space.g_theta_star_sa(c, r_sa, h_theta_sa);
            ActionPrediction {
                a_id,
                g_theta_sa,
                edge_id: None,
            }
        });
        self.predictions.extend(predictions);
        let new_predictions = &mut self.predictions[start..];
        new_predictions.sort_by(|a, b| b.a_id.cmp(&a.a_id));
        sample.sample_slice(new_predictions);
        let current_end = self.predictions.len();
        let target_end = start + sample.len();
        if current_end > target_end {
            self.predictions.truncate(target_end);
            node_weight.actions = (start as u32)..(target_end as u32);
        } else {
            node_weight.actions = (start as u32)..(current_end as u32);
        }
        debug_assert_ne!(start, self.predictions.len());
    }
}

#[derive(Clone)]
pub struct SamplePattern {
    pub max: usize,
    pub mid: usize,
    pub min: usize,
}

impl SamplePattern {
    fn sample_slice(&self, slice: &mut [ActionPrediction]) {
        if slice.len() > self.len() {
            let mid_and_min = &mut slice[self.max..];
            // pull the mid elements to the front from their current fencepost position
            for i in 0..self.mid {
                let mid_pos = Self::fencepost_position(i, self.mid, mid_and_min.len() - self.min);
                mid_and_min[i] = mid_and_min[mid_pos].clone();
            }
            // pull the min elements to the front from the back
            let tail = &mut mid_and_min[self.mid..];
            for i in 0..self.min {
                let min_pos = tail.len() - self.min + i;
                tail[i] = tail[min_pos].clone();
            }
        }
    }

    fn len(&self) -> usize {
        self.max + self.mid + self.min
    }

    fn fencepost_position(i: usize, k: usize, l: usize) -> usize {
        // rounds i * (l - 1) / (k - 1) to the nearest integer with only integer arithmetic
        (i * (l - 1) + (k - 1) / 2) / (k - 1)
    }
}