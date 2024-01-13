use super::axioms::{ActionOrderIndependent, ActionsNeverRepeat};

pub struct Layered<const LAYERS: usize, Space> {
    pub space: Space,
}

impl<const LAYERS: usize, Space> Layered<LAYERS, Space> {
    pub const fn new(space: Space) -> Self {
        Self { space }
    }
}

unsafe impl<const LAYERS: usize, Space: ActionsNeverRepeat> ActionsNeverRepeat
    for Layered<LAYERS, Space>
{
}
unsafe impl<const LAYERS: usize, Space: ActionOrderIndependent> ActionOrderIndependent
    for Layered<LAYERS, Space>
{
}
