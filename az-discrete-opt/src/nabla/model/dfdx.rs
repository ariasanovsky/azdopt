use dfdx::{tensor::{Cuda, Tensor, ZerosTensor}, nn::{BuildOnDevice, DeviceBuildExt, ZeroGrads}, prelude::{Gradients, Rank2}};

pub struct ActionModel<
    M,
    const BATCH: usize,
    const STATE: usize,
    const ACTION: usize,
> where
    M: BuildOnDevice<Cuda, f32>,
{
    dev: Cuda,
    model: <M as BuildOnDevice<Cuda, f32>>::Built,
    gradients: Option<Gradients<f32, Cuda>>,
    states_dev: Tensor<Rank2<BATCH, STATE>, f32, Cuda>,
    actions_dev: Tensor<Rank2<BATCH, ACTION>, f32, Cuda>,
}

impl<M, const BATCH: usize, const STATE: usize, const ACTION: usize>
    ActionModel<M, BATCH, STATE, ACTION>
where
    M: BuildOnDevice<Cuda, f32>,
{
    pub fn new(dev: Cuda) -> Self {
        let model = dev.build_module::<M, f32>();
        let gradients = Some(model.alloc_grads());
        let states_dev = dev.zeros();
        let actions_dev = dev.zeros();
        Self {
            dev,
            model,
            gradients,
            states_dev,
            actions_dev,
        }
    }
}