pub struct AHState<const N: usize, const E: usize>;

impl<const N: usize, const E: usize> AHState<N, E> {
    pub fn par_generate_batch<const BATCH: usize>() -> [Self; BATCH] {
        let _: () = crate::CheckFirstChooseTwoEqualsSecond::<N, E>::VALID;
        todo!()
    }
}
