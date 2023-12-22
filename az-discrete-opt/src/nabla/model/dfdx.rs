use dfdx::{tensor::{Cuda, Tensor, ZerosTensor}, nn::{BuildOnDevice, DeviceBuildExt, ZeroGrads}, prelude::{Gradients, Rank2}};

use dfdx::{
    losses::{cross_entropy_with_logits_loss, mse_loss},
    nn::{Module},
    optim::{Adam, Optimizer},
    shapes::Axis,
    tensor::{AsArray, OwnedTape, Trace},
    tensor_ops::{AdamConfig, Backward},
};

use super::NablaModel;

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
        let model = dev.build_module::<M, _>();
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


impl<M, const BATCH: usize, const STATE: usize, const ACTION: usize>
    NablaModel for ActionModel<M, BATCH, STATE, ACTION>
where
    M: BuildOnDevice<Cuda, f32>,
    <M as dfdx::nn::BuildOnDevice<dfdx::tensor::Cuda, f32>>::Built: Module<
        Tensor<Rank2<BATCH, STATE>, f32, Cuda>,
        Output = Tensor<Rank2<BATCH, ACTION>, f32, Cuda>,
    >,
    <M as dfdx::nn::BuildOnDevice<dfdx::tensor::Cuda, f32>>::Built: Module<
        Tensor<Rank2<BATCH, STATE>, f32, Cuda, OwnedTape<f32, Cuda>>,
        Output = Tensor<Rank2<BATCH, ACTION>, f32, Cuda, OwnedTape<f32, Cuda>>,
    >,
{
    fn write_predictions(&mut self, x_t: &[f32], predictions: &mut [f32]) {
        let Self {
            dev,
            model,
            gradients,
            states_dev,
            actions_dev,
        } = self;
        debug_assert_eq!(x_t.len(), BATCH * STATE);
        debug_assert_eq!(predictions.len(), BATCH * ACTION);
        states_dev.copy_from(x_t);
        *actions_dev = model.forward(states_dev.clone());
        actions_dev.copy_into(predictions);
    }

    fn update_model(&mut self, x_t: &[f32], observations: &[f32]) -> f32 {
        todo!()
    }
}