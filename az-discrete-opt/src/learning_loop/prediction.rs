pub struct PredictionData<const BATCH: usize, const ACTION: usize, const GAIN: usize> {
    pi: [[f32; ACTION]; BATCH],
    g: [[f32; GAIN]; BATCH],
}

impl<const BATCH: usize, const ACTION: usize, const GAIN: usize> Default
    for PredictionData<BATCH, ACTION, GAIN>
{
    fn default() -> Self {
        Self {
            pi: [[0.; ACTION]; BATCH],
            g: [[0.; GAIN]; BATCH],
        }
    }
}

impl<const BATCH: usize, const ACTION: usize, const GAIN: usize>
    PredictionData<BATCH, ACTION, GAIN>
{
    pub fn get_mut(&mut self) -> (&mut [[f32; ACTION]; BATCH], &mut [[f32; GAIN]; BATCH]) {
        (&mut self.pi, &mut self.g)
    }

    pub fn get(&self) -> (&[[f32; ACTION]; BATCH], &[[f32; GAIN]; BATCH]) {
        (&self.pi, &self.g)
    }

    pub fn pi_mut(&mut self) -> &mut [[f32; ACTION]; BATCH] {
        &mut self.pi
    }

    pub fn g_mut(&mut self) -> &mut [[f32; GAIN]; BATCH] {
        &mut self.g
    }
}
