pub trait Log {
    type T;
    fn add_transition_data(
        &mut self,
        a1: usize,
        r1: f32,
        transition: &Self::T,
        end: FinalStateData,
    );
}

pub enum FinalStateData {
    Leaf,
    New { final_reward: f32 },
}
