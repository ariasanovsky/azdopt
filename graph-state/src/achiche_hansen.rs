pub struct AHState<const N: usize, const E: usize>;

impl<const N: usize, const E: usize> AHState<N, E> {
    pub fn par_generate_batch<const BATCH: usize>() -> [Self; BATCH] {
        crate::static_assert_first_choose_two_equals_second!(N, E);
        todo!()
    }
}
