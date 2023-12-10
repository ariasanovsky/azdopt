use crate::learning_loop::prediction::PredictionData;

#[cfg(feature = "tensorboard")]
use crate::tensorboard::Summarize;

#[cfg(feature = "candle")]
pub mod candle;
#[cfg(feature = "dfdx")]
pub mod dfdx;

#[cfg(feature = "rand_noise")]
pub fn add_dirichlet_noise(rng: &mut impl rand::Rng, p: &mut [f32], alpha: &[f32], epsilon: f32) {
    use rand_distr::Distribution;
    let dir = rand_distr::Dirichlet::new(alpha).unwrap();
    let sample = dir.sample(rng);
    p.iter_mut().zip(sample.iter()).for_each(|(p, dir)| {
        *p *= 1. - epsilon;
        *p += epsilon * dir;
    });
}

pub trait AzModel<const BATCH: usize, const STATE: usize, const ACTION: usize, const GAIN: usize> {
    // todo! fallible
    fn write_predictions(
        &mut self,
        x_t: &[[f32; STATE]; BATCH],
        predictions: &mut PredictionData<BATCH, ACTION, GAIN>,
    );

    // todo! fallible
    fn update_model(
        &mut self,
        x_t: &[[f32; STATE]; BATCH],
        observations: &PredictionData<BATCH, ACTION, GAIN>,
    ) -> Loss;
}

pub struct Loss {
    pub entropy: f32,
    pub mse: f32,
}

#[cfg(feature = "tensorboard")]
impl Summarize for Loss {
    fn summary(&self) -> tensorboard_writer::proto::tensorboard::Summary {
        tensorboard_writer::SummaryBuilder::new()
            .scalar("loss/entropy", self.entropy as _)
            .scalar("loss/mse", self.mse as _)
            .build()
    }
}
