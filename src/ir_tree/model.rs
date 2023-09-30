pub trait Model<S> {
    type P;
    type O;
    type L;
    fn predict(&self, state: &S) -> Self::P;
    fn update(&mut self, observations: Vec<Self::O>) -> Self::L;
}
