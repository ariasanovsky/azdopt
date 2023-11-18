// const C_PUCT: f32 = 100.0;

#[derive(Debug, PartialEq)]
pub(crate) struct INTVisitedActionData {
    pub(crate) a: usize,
    pub(crate) p_sa: f32,
    pub(crate) n_sa: usize,
    pub(crate) g_sa_sum: f32,
    pub(crate) u_sa: f32,
}

#[derive(Debug, PartialEq)]
pub(crate) struct INTUnvisitedActionData {
    pub(crate) a: usize,
    pub(crate) p_sa: f32,
}

impl INTVisitedActionData {
    pub(crate) fn update(&mut self, g_star_theta_i: f32) {
        let Self {
            a: _,
            p_sa: _,
            n_sa,
            g_sa_sum,
            u_sa: _,
        } = self;
        *n_sa += 1;
        *g_sa_sum += g_star_theta_i;
    }
}
