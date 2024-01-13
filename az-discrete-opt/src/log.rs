pub struct ArgminData<S, C> {
    pub state: S,
    pub cost: C,
    pub eval: f32,
}

impl<S, C> ArgminData<S, C> {
    pub fn new(state: S, cost: C, eval: f32) -> Self {
        Self { state, cost, eval }
    }
}
