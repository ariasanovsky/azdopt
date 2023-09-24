pub trait Log {
    type R;
    type T;
    type G;
    fn add_transition_data(
        &mut self,
        a1: usize,
        r1: Self::R,
        transition: &Self::T,
        end: FinalStateData<Self::G>,
    );
}

pub enum FinalStateData<G> {
    Leaf,
    New(G),
}
