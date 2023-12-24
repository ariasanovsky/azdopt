use crate::nabla::space::NablaStateActionSpace;

use super::{NewNodeKind, Transition, node::StateNode};

impl<P> NewNodeKind<'_, P> {
    pub fn update_existing_nodes<Space: NablaStateActionSpace>(
        self,
        space: &Space,
        transitions: Vec<Transition>,
        new_node: Option<StateNode>,
    ) {
        let mut c_s_t_theta_star = match (&self, &new_node) {
            (Self::OldExhaustedNode { c_s_t_theta_star }, _) => *c_s_t_theta_star,
            _ => todo!(),
        };
        transitions.into_iter().rev().for_each(|t| {
            let Transition {
                c_s,
                g_theta_star_sa,
                kind_sa,
            } = t;
            todo!("update stuff");
            c_s_t_theta_star = c_s_t_theta_star.min(c_s);
        });
    }
}