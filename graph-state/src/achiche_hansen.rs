use az_discrete_opt::state::Cost;

#[derive(Clone)]
pub struct AHState<const N: usize, const E: usize>;

impl<const N: usize, const E: usize> AHState<N, E> {
    pub fn par_generate_batch<const BATCH: usize, const STATE: usize>() -> ([Self; BATCH], [[f32; STATE]; BATCH]) {
        let _: () = crate::CheckFirstChooseTwoEqualsSecond::<N, E>::VALID;
        let _: () = crate::CheckFirstTimesThreePlusOneEqualsSecond::<E, STATE>::VALID;
        todo!()
    }
}

impl<const N: usize, const E: usize> Cost for AHState<N, E> {
    fn cost(&self) -> f32 {
        todo!()
    }
}