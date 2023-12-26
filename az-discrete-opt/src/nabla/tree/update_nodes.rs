use crate::nabla::space::NablaStateActionSpace;

use super::{NodeKind, Transition, node::{StateNode, TransitionKind}};

impl<P> NodeKind<'_, P> {
    pub fn update_existing_nodes<Space: NablaStateActionSpace>(
        self,
        space: &Space,
        state: &Space::State,
        cost: &Space::Cost,
        h_theta: &[f32],
        path: &P,
        max_num_actions: usize,
        transitions: Vec<Transition>,
    ) -> Option<StateNode>
    where
        P: Ord + Clone,
    {
        let (node, mut kind): (_, TransitionKind) = match self {
            NodeKind::NewLevel => {
                let (n, kind) = StateNode::new(space, state, cost, h_theta, max_num_actions);
                (Some(n), kind)
            },
            NodeKind::New(level) => {
                let (n, kind)= StateNode::new(space, state, cost, h_theta, max_num_actions);
                level.insert(path.clone(), n);
                (None, kind)
            },
            NodeKind::OldExhausted { c_s_t_theta_star } => (
                None,
                TransitionKind::Exhausting { c_theta_star: c_s_t_theta_star },
            ),
        };
        // let (mut c_theta_star, mut exhausting) = todo!();
        // match (&self, &new_node) {
        //     (Self::OldExhausted { c_s_t_theta_star }, _) => (*c_s_t_theta_star, TransitionKind::Exhausting),
        //     (_, Some(new_node)) => (new_node.cost(), TransitionKind::Active),
        //     _ => unreachable!(),
        // };
        transitions.into_iter().rev().for_each(|t| {
            t.update_action_data(&mut kind);
        });
        node
    }
}