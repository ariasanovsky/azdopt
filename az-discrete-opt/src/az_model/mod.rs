use rand_distr::Distribution;

pub fn add_dirichlet_noise(
    rng: &mut impl rand::Rng,
    p: &mut [f32],
    alpha: &[f32],
    epsilon: f32,
) {
    let dir = rand_distr::Dirichlet::new(alpha).unwrap();
    let sample = dir.sample(rng);
    p.iter_mut().zip(sample.into_iter()).for_each(|(p, dir)| {
        *p *= 1. - epsilon;
        *p += epsilon * dir;
    });
}
