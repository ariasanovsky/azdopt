use dfdx::{
    nn::{BuildOnDevice, DeviceBuildExt, ZeroGrads},
    prelude::{Gradients, Rank2},
    shapes::{Rank0, Rank1},
    tensor::{Cuda, Tensor, ZerosTensor},
    tensor_ops::SumTo,
};

use dfdx::{
    nn::Module,
    optim::{Adam, Optimizer},
    tensor::{AsArray, OwnedTape, Trace},
    tensor_ops::{AdamConfig, Backward},
};

use super::NablaModel;

pub struct ActionModel<M, const BATCH: usize, const STATE: usize, const ACTION: usize>
where
    M: BuildOnDevice<Cuda, f32>,
{
    model: <M as BuildOnDevice<Cuda, f32>>::Built,
    gradients: Option<Gradients<f32, Cuda>>,
    optimizer: Adam<<M as BuildOnDevice<Cuda, f32>>::Built, f32, Cuda>,
    states_dev: Tensor<Rank2<BATCH, STATE>, f32, Cuda>,
    action_weights_dev: Tensor<Rank2<BATCH, ACTION>, f32, Cuda>,
    // state_weights_dev: Tensor<Rank1<BATCH>, f32, Cuda>,
    actions_dev: Tensor<Rank2<BATCH, ACTION>, f32, Cuda>,
}

impl<M, const BATCH: usize, const STATE: usize, const ACTION: usize>
    ActionModel<M, BATCH, STATE, ACTION>
where
    M: BuildOnDevice<Cuda, f32>,
{
    pub fn new(dev: Cuda, cfg: AdamConfig) -> Self {
        let model = dev.build_module::<M, _>();
        let gradients = Some(model.alloc_grads());
        let states_dev = dev.zeros();
        let action_weights_dev = dev.zeros();
        // let state_weights_dev = dev.zeros();
        let actions_dev = dev.zeros();
        let optimizer = Adam::new(&model, cfg);
        Self {
            model,
            gradients,
            optimizer,
            states_dev,
            action_weights_dev,
            // state_weights_dev,
            actions_dev,
        }
    }
}

impl<M, const BATCH: usize, const STATE: usize, const ACTION: usize> NablaModel
    for ActionModel<M, BATCH, STATE, ACTION>
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
    fn write_predictions(&mut self, states: &[f32], predictions: &mut [f32]) {
        let Self {
            model,
            gradients: _,
            optimizer: _,
            states_dev,
            action_weights_dev: _,
            // state_weights_dev: _,
            actions_dev,
        } = self;
        debug_assert_eq!(states.len(), BATCH * STATE);
        debug_assert_eq!(predictions.len(), BATCH * ACTION);
        states_dev.copy_from(states);
        *actions_dev = model.forward(states_dev.clone());
        actions_dev.copy_into(predictions);
    }

    fn update_model(
        &mut self,
        states: &[f32],
        observations: &[f32],
        action_weights: &[f32],
    ) -> f32 {
        let Self {
            model,
            gradients,
            optimizer,
            states_dev,
            action_weights_dev,
            actions_dev,
        } = self;
        debug_assert_eq!(states.len(), BATCH * STATE);
        debug_assert_eq!(observations.len(), BATCH * ACTION);
        debug_assert_eq!(action_weights.len(), BATCH * ACTION);
        states_dev.copy_from(states);
        actions_dev.copy_from(observations);
        action_weights_dev.copy_from(action_weights);
        let weight_sum = action_weights.iter().sum::<f32>();
        dbg!(weight_sum);
        // let action_normalization = action_weights_dev.clone().sum::<Rank1<BATCH>, _>().recip();
        // let action_normalization: Tensor<Rank2<BATCH, ACTION>, _, _> = action_normalization.broadcast();
        *action_weights_dev = action_weights_dev.clone() / weight_sum;
        // let state_normalization = state_weights_dev.clone().sum::<Rank0, _>().recip();
        // let state_normalization: Tensor<Rank1<BATCH>, _, _> = state_normalization.broadcast();
        // *state_weights_dev = state_weights_dev.clone() * state_normalization;
        let gradients = gradients.take().unwrap_or_else(|| model.alloc_grads());
        let states_traced = states_dev.clone().trace(gradients);
        let predictions = model.forward(states_traced);

        let error = (predictions - actions_dev.clone()).square();
        let loss = (
            (error * action_weights_dev.clone()).sum::<Rank1<BATCH>, _>()
            // * state_weights_dev.clone()
        )
        .sum::<Rank0, _>();
        let l = loss.array();
        dbg!(l);
        let mut grads = loss.backward();
        optimizer.update(model, &mut grads).unwrap();
        model.zero_grads(&mut grads);
        self.gradients = Some(grads);
        l
    }
}
