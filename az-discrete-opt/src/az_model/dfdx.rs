use dfdx::{
    losses::{cross_entropy_with_logits_loss, mse_loss},
    nn::{BuildOnDevice, DeviceBuildExt, Module, ZeroGrads},
    optim::{Adam, Optimizer},
    shapes::Axis,
    tensor::{AsArray, Cuda, OwnedTape, Tensor, Trace, ZerosTensor},
    tensor_ops::{AdamConfig, Backward},
};

use dfdx::prelude::{Gradients, Rank2};

use crate::learning_loop::prediction::PredictionData;

use super::{AzModel, Loss};

pub struct TwoModels<
    L,
    G,
    const BATCH: usize,
    const STATE: usize,
    const ACTION: usize,
    const GAIN: usize,
> where
    L: BuildOnDevice<Cuda, f32>,
    G: BuildOnDevice<Cuda, f32>,
{
    dev: Cuda,
    pi_model: <L as BuildOnDevice<Cuda, f32>>::Built,
    g_model: <G as BuildOnDevice<Cuda, f32>>::Built,
    pi_adam: Adam<<L as BuildOnDevice<Cuda, f32>>::Built, f32, Cuda>,
    g_adam: Adam<<G as BuildOnDevice<Cuda, f32>>::Built, f32, Cuda>,
    pi_gradients: Option<Gradients<f32, Cuda>>,
    g_gradients: Option<Gradients<f32, Cuda>>,
    x_t_dev: Tensor<Rank2<BATCH, STATE>, f32, Cuda>,
    pi_t_dev: Tensor<Rank2<BATCH, ACTION>, f32, Cuda>,
    g_t_dev: Tensor<Rank2<BATCH, GAIN>, f32, Cuda>,
    logits_mask_dev: Option<Tensor<Rank2<BATCH, ACTION>, f32, Cuda>>,
}

impl<L, G, const BATCH: usize, const STATE: usize, const ACTION: usize, const GAIN: usize>
    TwoModels<L, G, BATCH, STATE, ACTION, GAIN>
where
    L: BuildOnDevice<Cuda, f32>,
    G: BuildOnDevice<Cuda, f32>,
{
    pub fn new(dev: Cuda, pi_config: AdamConfig, g_config: AdamConfig) -> Self {
        let pi_model = dev.build_module::<L, f32>();
        let g_model = dev.build_module::<G, f32>();
        let pi_adam = Adam::new(&pi_model, pi_config);
        let g_adam = Adam::new(&g_model, g_config);
        let pi_gradients = Some(pi_model.alloc_grads());
        let g_gradients = Some(g_model.alloc_grads());
        let x_t_dev = dev.zeros();
        let pi_t_dev = dev.zeros();
        let g_t_dev = dev.zeros();
        let logits_mask_dev = None;
        Self {
            dev,
            pi_model,
            g_model,
            pi_adam,
            g_adam,
            pi_gradients,
            g_gradients,
            x_t_dev,
            pi_t_dev,
            g_t_dev,
            logits_mask_dev,
        }
    }
}

impl<L, G, const BATCH: usize, const STATE: usize, const ACTION: usize, const GAIN: usize>
    AzModel for TwoModels<L, G, BATCH, STATE, ACTION, GAIN>
where
    L: BuildOnDevice<Cuda, f32>,
    G: BuildOnDevice<Cuda, f32>,
    <L as dfdx::nn::BuildOnDevice<dfdx::tensor::Cuda, f32>>::Built: Module<
        Tensor<Rank2<BATCH, STATE>, f32, Cuda>,
        Output = Tensor<Rank2<BATCH, ACTION>, f32, Cuda>,
    >,
    <G as dfdx::nn::BuildOnDevice<dfdx::tensor::Cuda, f32>>::Built: Module<
        Tensor<Rank2<BATCH, STATE>, f32, Cuda>,
        Output = Tensor<Rank2<BATCH, GAIN>, f32, Cuda>,
    >,
    <L as dfdx::nn::BuildOnDevice<dfdx::tensor::Cuda, f32>>::Built: Module<
        Tensor<Rank2<BATCH, STATE>, f32, Cuda, OwnedTape<f32, Cuda>>,
        Output = Tensor<Rank2<BATCH, ACTION>, f32, Cuda, OwnedTape<f32, Cuda>>,
    >,
    <G as dfdx::nn::BuildOnDevice<dfdx::tensor::Cuda, f32>>::Built: Module<
        Tensor<Rank2<BATCH, STATE>, f32, Cuda, OwnedTape<f32, Cuda>>,
        Output = Tensor<Rank2<BATCH, GAIN>, f32, Cuda, OwnedTape<f32, Cuda>>,
    >,
{
    fn write_predictions(
        &mut self,
        x_t: &[f32],
        // logits_mask: Option<&[f32]>,
        predictions: &mut PredictionData,
    ) {
        let Self {
            dev,
            pi_model,
            g_model,
            pi_adam: _,
            g_adam: _,
            pi_gradients: _,
            g_gradients: _,
            x_t_dev,
            pi_t_dev,
            g_t_dev,
            logits_mask_dev,
        } = self;
        let (pi_t_theta, g_t_theta) = predictions.get_mut();
        x_t_dev.copy_from(x_t);
        
        // *pi_t_dev = if let Some(mask) = logits_mask {
        //     let mask_dev = logits_mask_dev.get_or_insert_with(|| dev.zeros());
        //     mask_dev.copy_from(mask);
        //     pi_model.forward(x_t_dev.clone()) + mask_dev.clone()
        // } else {
        //     pi_model.forward(x_t_dev.clone())
        // }.softmax::<Axis<1>>();
        *pi_t_dev = pi_model.forward(x_t_dev.clone()).softmax::<Axis<1>>();
        
        pi_t_dev.copy_into(pi_t_theta);
        *g_t_dev = g_model.forward(x_t_dev.clone());
        g_t_dev.copy_into(g_t_theta);
    }

    fn update_model(
        &mut self,
        x_t: &[f32],
        logits_mask: Option<&[f32]>,
        observations: &PredictionData,
    ) -> Loss {
        let Self {
            dev,
            pi_model,
            g_model,
            pi_adam,
            g_adam,
            pi_gradients,
            g_gradients,
            x_t_dev,
            pi_t_dev,
            g_t_dev,
            logits_mask_dev,
        } = self;

        let (pi_0_obs, g_0_obs) = observations.get();

        // update probability predictions
        x_t_dev.copy_from(x_t);
        pi_t_dev.copy_from(pi_0_obs);
        let mut some_pi_gradients = pi_gradients
            .take()
            .unwrap_or_else(|| pi_model.alloc_grads());
        let mut predicted_logits_traced = pi_model.forward(x_t_dev.clone().traced(some_pi_gradients));
        if let Some(mask) = logits_mask {
            let mask_dev = logits_mask_dev.get_or_insert_with(|| dev.zeros());
            mask_dev.copy_from(mask);
            predicted_logits_traced = predicted_logits_traced + mask_dev.clone();
        }
        let cross_entropy =
            cross_entropy_with_logits_loss(predicted_logits_traced, pi_t_dev.clone());
        let entropy = cross_entropy.array();
        some_pi_gradients = cross_entropy.backward();
        pi_adam
            .update(pi_model, &some_pi_gradients)
            .expect("optimizer failed");
        pi_model.zero_grads(&mut some_pi_gradients);
        *pi_gradients = Some(some_pi_gradients);

        // update mean max gain prediction
        g_t_dev.copy_from(g_0_obs);
        let mut some_g_gradients = g_gradients.take().unwrap_or_else(|| g_model.alloc_grads());
        let predicted_values_traced = g_model.forward(x_t_dev.clone().traced(some_g_gradients));
        let g_loss = mse_loss(predicted_values_traced, g_t_dev.clone());
        let mse = g_loss.array();
        some_g_gradients = g_loss.backward();
        g_adam
            .update(g_model, &some_g_gradients)
            .expect("optimizer failed");
        g_model.zero_grads(&mut some_g_gradients);
        *g_gradients = Some(some_g_gradients);
        Loss { entropy, mse }
    }
}
