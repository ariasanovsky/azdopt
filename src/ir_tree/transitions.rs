pub struct Transitions<P, S> {
    pub a1: usize,
    pub r1: f32,
    pub transitions: Vec<(P, usize, f32)>,
    pub end: FinalState<P, S>,
}

pub enum FinalState<P, S> {
    Leaf(P, S),
    New(P, S),
}
