pub struct Transitions<R, P, S> {
    pub a1: usize,
    pub r1: R,
    pub transitions: Vec<(P, usize, R)>,
    pub end: FinalState<P, S>,
}

pub enum FinalState<P, S> {
    Leaf(P, S),
    New(P, S),
}
