use crate::nabla::space::NablaStateActionSpace;

use super::{NewNodeKind, Transition, node::{StateNode, TransitionKind}};

impl<P> NewNodeKind<'_, P> {
    pub fn update_existing_nodes<Space: NablaStateActionSpace>(
        self,
        space: &Space,
        transitions: Vec<Transition>,
        new_node: Option<StateNode>,
    ) {
        let (mut c_theta_star, mut exhausting) = match (&self, &new_node) {
            (Self::OldExhaustedNode { c_s_t_theta_star }, _) => (*c_s_t_theta_star, TransitionKind::Exhausting),
            (_, Some(new_node)) => (new_node.cost(), TransitionKind::Active),
            _ => unreachable!(),
        };
        transitions.into_iter().rev().for_each(|t| {
            t.update_action_data(&mut exhausting, &mut c_theta_star);
        });
    }
}