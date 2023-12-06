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
        pi_t_theta: &mut [[f32; ACTION]; BATCH],
        g_t_theta: &mut [[f32; GAIN]; BATCH],
    );

    fn update_model(
        &mut self,
        x_t: &[[f32; STATE]; BATCH],
        pi_0_obs: &[[f32; ACTION]; BATCH],
        g_0_obs: &[[f32; GAIN]; BATCH],
    ) -> (f32, f32);
}