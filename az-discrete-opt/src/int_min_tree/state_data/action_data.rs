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
    pub(crate) fn new(a: usize, p_a: f32) -> Self {
        todo!();
        // Self {
        //     a,
        //     p_sa: p_a,
        //     n_sa: 0,
        //     g_sa_sum: 0.0,
        //     u_sa: C_PUCT * p_a,
        // }
    }

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

    // pub(crate) fn update_upper_estimate(&mut self, n_s: usize) {
    //     let Self {
    //         a: _,
    //         p_sa,
    //         n_sa,
    //         g_sa_sum,
    //         u_sa,
    //     } = self;
    //     debug_assert_ne!(n_s, 0);
    //     let n_sa = (*n_sa + 1) as f32;
    //     let g_sa = *g_sa_sum / n_sa;
    //     let n_s = n_s as f32;
    //     let p_sa = *p_sa;
    //     *u_sa = g_sa + C_PUCT * p_sa * (n_s.sqrt() / n_sa);
    //     println!(
    //         "{u_sa} = {g_sa_sum} / {n_sa} + {C_PUCT} * {p_sa} * ({n_s}.sqrt() / {n_sa})",
    //     );
    //     if true {

    //     }
    // }
}
