use crate::nabla::space::NablaStateActionSpace;

pub(super) struct StateNode {
    // n_s: usize,
    // actions: Vec
}

pub(super) enum StateNodeKind {
    // Active { node: StateNode },
    // Exhausted { c_star: f32 },
}

impl StateNodeKind {
    pub(super) fn new<Space: NablaStateActionSpace>(
        space: &Space,
        s: Space::State,
        c: &Space::Cost,
        h_theta: &[f32],
    ) -> Self {
        todo!()
    }
}