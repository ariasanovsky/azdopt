use candle_core::{Device, Module, Tensor};
use candle_nn::ops::softmax;

use super::AzModel;

pub struct TwoModels<P, G> {
    device: Device,
    pi_model: P,
    g_model: G,
    x_t_dev: Tensor,
    pi_t_dev: Tensor,
    g_t_dev: Tensor,
}

impl<P, G, const BATCH: usize, const STATE: usize, const ACTION: usize, const GAIN: usize>
    AzModel<BATCH, STATE, ACTION, GAIN> for TwoModels<P, G>
where
    P: Module,
    G: Module,
{
    fn write_predictions(
        &mut self,
        x_t: &[[f32; STATE]; BATCH],
        predictions: &mut crate::learning_loop::prediction::PredictionData<BATCH, ACTION, GAIN>,
    ) {
        let Self {
            device,
            pi_model,
            g_model,
            x_t_dev,
            pi_t_dev,
            g_t_dev,
        } = self;
        let (pi_t_theta, g_t_theta) = predictions.get_mut();
        // allocation?
        *x_t_dev = Tensor::from_slice(x_t.flatten(), (BATCH, STATE), device).unwrap();
        *pi_t_dev = pi_model.forward(&x_t_dev).unwrap();
        *pi_t_dev = softmax(pi_t_dev, 1).unwrap();
        // https://docs.rs/candle-core/0.3.1/src/candle_core/convert.rs.html#117
        todo!()
    }

    fn update_model(
        &mut self,
        x_t: &[[f32; STATE]; BATCH],
        logits_mask: Option<&[[f32; ACTION]; BATCH]>,
        observations: &crate::learning_loop::prediction::PredictionData<BATCH, ACTION, GAIN>,
    ) -> super::Loss {
        todo!()
    }
}
