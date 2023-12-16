use candle_core::{Device, Module, Tensor};
use candle_nn::ops::softmax;

use crate::learning_loop::prediction::PredictionData;

use super::AzModel;

pub struct TwoModels<P, G> {
    device: Device,
    pi_model: P,
    g_model: G,
    x_t_dev: Tensor,
    pi_t_dev: Tensor,
    g_t_dev: Tensor,
    batch: usize,
    state: usize,
}

impl<P, G>
    AzModel for TwoModels<P, G>
where
    P: Module,
    G: Module,
{
    fn write_predictions(
        &mut self,
        x_t: &[f32],
        // logits_mask: Option<&[f32]>,
        predictions: &mut PredictionData,
    ) {
        let Self {
            device,
            pi_model,
            g_model,
            x_t_dev,
            pi_t_dev,
            g_t_dev,
            batch,
            state,
        } = self;
        let (pi_t_theta, g_t_theta) = predictions.get_mut();
        // allocation?
        *x_t_dev = Tensor::from_slice(x_t, (*batch, *state), device).unwrap();
        *pi_t_dev = pi_model.forward(&x_t_dev).unwrap();
        *pi_t_dev = softmax(pi_t_dev, 1).unwrap();
        // https://docs.rs/candle-core/0.3.1/src/candle_core/convert.rs.html#117
        todo!()
    }

    fn update_model(
        &mut self,
        x_t: &[f32],
        logits_mask: Option<&[f32]>,
        observations: &PredictionData,
    ) -> super::Loss {
        todo!()
    }
}
