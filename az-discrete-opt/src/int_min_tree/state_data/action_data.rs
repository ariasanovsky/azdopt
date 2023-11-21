// const C_PUCT: f32 = 100.0;

use super::UpperEstimateData;

#[derive(Debug, PartialEq)]
pub(crate) struct INTVisitedActionData {
    a: usize,
    p_sa: f32,
    n_sa: usize,
    g_sa_sum: f32,
    u_sa: f32,
}

#[derive(Debug, PartialEq)]
pub(crate) struct INTUnvisitedActionData {
    a: usize,
    p_sa: f32,
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

    pub fn action(&self) -> usize {
        self.a
    }

    pub(crate) fn n_sa(&self) -> usize {
        self.n_sa
    }

    pub(crate) fn g_sa(&self) -> f32 {
        debug_assert_ne!(self.n_sa, 0);
        self.g_sa_sum / self.n_sa as f32
    }

    pub(crate) fn set_upper_estimate(&mut self, u_sa: f32) {
        self.u_sa = u_sa;
    }

    pub(crate) fn upper_estimate_data(&self, n_s: usize, depth: usize) -> UpperEstimateData {
        UpperEstimateData {
            n_s,
            n_sa: self.n_sa,
            g_sa_sum: self.g_sa_sum,
            p_sa: self.p_sa,
            depth,
        }
    }

    pub(crate) fn upper_estimate(&self) -> f32 {
        self.u_sa
    }
}

impl INTUnvisitedActionData {
    pub(crate) fn action(&self) -> usize {
        self.a
    }

    pub(crate) fn p_sa(&self) -> f32 {
        self.p_sa
    }

    pub(crate) fn  to_visited_action(self, g_star_theta_i: f32) -> INTVisitedActionData {
        let INTUnvisitedActionData { a, p_sa } = self;
        INTVisitedActionData {
            a,
            p_sa,
            n_sa: 1,
            g_sa_sum: g_star_theta_i,
            u_sa: 0.0,
        }
    }

    pub(crate) fn new(a: usize, p_sa: f32) -> Self {
        Self { a, p_sa }
    }

    pub(crate) fn upper_estimate_data(&self, n_s: usize, depth: usize) -> UpperEstimateData {
        UpperEstimateData {
            n_s,
            n_sa: 1,
            g_sa_sum: 0.,
            p_sa: self.p_sa,
            depth,
        }
    }
}
