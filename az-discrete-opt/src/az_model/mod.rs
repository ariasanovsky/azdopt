use crate::learning_loop::prediction::PredictionData;

pub mod dfdx;
pub mod candle;

pub fn add_dirichlet_noise(
    rng: &mut impl rand::Rng,
    p: &mut [f32],
    alpha: &[f32],
    epsilon: f32,
) {
    use rand_distr::Distribution;
    let dir = rand_distr::Dirichlet::new(alpha).unwrap();
    let sample = dir.sample(rng);
    p.iter_mut().zip(sample.into_iter()).for_each(|(p, dir)| {
        *p *= 1. - epsilon;
        *p += epsilon * dir;
    });
}

pub trait AzModel<const BATCH: usize, const STATE: usize, const ACTION: usize, const GAIN: usize> {
    fn write_predictions(
        &mut self,
        x_t: &[[f32; STATE]; BATCH],
        predictions: &mut PredictionData<BATCH, ACTION, GAIN>,
    );

    fn update_model(
        &mut self,
        x_t: &[[f32; STATE]; BATCH],
        observations: &PredictionData<BATCH, ACTION, GAIN>,
    ) -> (f32, f32);
}
