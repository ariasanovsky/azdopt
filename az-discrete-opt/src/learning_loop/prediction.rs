pub struct PredictionData<'a> {
    pi: &'a mut [f32],
    g: &'a mut [f32],
}

impl<'a> PredictionData<'a> {
    pub fn new(pi: &'a mut [f32], g: &'a mut [f32]) -> Self {
        Self { pi, g }
    }

    pub fn get_mut(&mut self) -> (&mut [f32], &mut [f32]) {
        (self.pi, self.g)
    }

    pub fn get(&self) -> (&[f32], &[f32]) {
        (self.pi, self.g)
    }

    pub fn pi_mut(&mut self) -> &mut [f32] {
        self.pi
    }

    pub fn g_mut(&mut self) -> &mut [f32] {
        self.g
    }
}
