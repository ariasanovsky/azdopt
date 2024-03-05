#[cfg(feature = "dfdx")]
pub mod dfdx;

pub trait NablaModel {
    fn write_predictions(
        &mut self,
        states: &[f32],
        valid_actions: &[bool],
        num_actions: &[f32],
        v_predictions: &mut [f32],
        p_predictions: &mut [f32]
    );
    fn update_model(
        &mut self,
        states: &[f32],
        valid_actions: &[bool],
        num_actions: &[f32],
        v_observations: &[f32],
        n_observations: &[f32],
    )
        -> f32;
}

pub struct TrivialModel;

impl NablaModel for TrivialModel {
    fn write_predictions(
        &mut self,
        _states: &[f32],
        _valid_actions: &[bool],
        _num_actions: &[f32],
        _v_predictions: &mut [f32],
        _p_predictions: &mut [f32],
    ) {}

    fn update_model(
        &mut self,
        _states: &[f32],
        _valid_actions: &[bool],
        _num_actions: &[f32],
        _v_predictions: &[f32],
        _n_observations: &[f32],
    ) -> f32 {
        0.
    }
}

#[cfg(feature = "tensorboard")]
impl crate::tensorboard::Summarize for f32 {
    fn summary(&self) -> tensorboard_writer::proto::tensorboard::Summary {
        tensorboard_writer::SummaryBuilder::new()
            .scalar("loss", *self as _)
            .build()
    }
}
