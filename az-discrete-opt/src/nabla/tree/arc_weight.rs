use crate::nabla::tree::EdgeIndex;

#[derive(Debug)]
pub struct ActionWeight {
    // pub(crate) g_t_sa: f32,
    // pub(crate) h_t_sa: f32,
    pub(crate) prediction_pos: usize,
}

#[derive(Debug, Clone)]
pub struct ActionPrediction {
    pub(crate) a_id: usize,
    // pub(crate) h_theta_sa: f32,
    pub(crate) g_theta_sa: f32,
    pub(crate) edge_id: Option<EdgeIndex>,
}
