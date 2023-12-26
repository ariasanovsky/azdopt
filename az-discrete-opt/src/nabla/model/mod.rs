#[cfg(feature = "dfdx")]
pub mod dfdx;

pub trait NablaModel {
    fn write_predictions(&mut self, states: &[f32], predictions: &mut [f32]);
    fn update_model(
        &mut self,
        states: &[f32],
        observations: &[f32],
        weights: &[f32],
    ) -> f32;
}

#[cfg(feature = "tensorboard")]
impl crate::tensorboard::Summarize for f32 {
    fn summary(&self) -> tensorboard_writer::proto::tensorboard::Summary {
        tensorboard_writer::SummaryBuilder::new()
            .scalar("loss", *self as _)
            .build()
    }
}
