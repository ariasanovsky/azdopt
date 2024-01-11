use crate::space::StateActionSpace;

// ?todo!
// pub struct StateData<'a, S, C> {
//     roots: *mut S,
//     states: *mut S,
//     costs: *mut C,
//     vectors: *mut f32,
//     batch: usize,
//     dim: usize,
//     _marker: std::marker::PhantomData<&'a mut ()>,
// }

pub struct StateData<'a, S, C> {
    roots: &'a mut [S],
    states: &'a mut [S],
    costs: &'a mut [C],
    vectors: &'a mut [f32],
}

impl<'a, S, C> StateData<'a, S, C> {
    pub fn new(
        roots: &'a mut [S],
        states: &'a mut [S],
        costs: &'a mut [C],
        vectors: &'a mut [f32],
        dim: usize,
    ) -> Self {
        let batch = roots.len();
        assert_eq!(batch, states.len());
        assert_eq!(batch, costs.len());
        assert_eq!(batch, vectors.len() / dim);
        Self {
            roots,
            states,
            costs,
            vectors,
        }
    }

    pub fn get_states(&self) -> &[S] {
        self.states
    }

    pub fn get_states_mut(&mut self) -> &mut [S] {
        self.states
    }

    pub fn get_costs(&self) -> &[C] {
        self.costs
    }

    pub fn get_vectors(&self) -> &[f32] {
        self.vectors
    }

    pub fn reset_states(&mut self)
    where
        S: Clone,
    {
        self.states.clone_from_slice(self.roots);
    }

    pub fn get_roots(&self) -> &[S] {
        self.roots
    }

    pub fn get_roots_mut(&mut self) -> &mut [S] {
        self.roots
    }

    #[cfg(feature = "rayon")]
    pub fn par_write_state_vecs<Space>(&mut self, space: &Space)
    where
        Space: StateActionSpace<State = S> + Sync,
        S: Sync,
    {
        use rayon::{
            iter::{IntoParallelIterator, ParallelIterator},
            slice::ParallelSliceMut,
        };

        (
            self.states as &_,
            self.vectors.par_chunks_exact_mut(Space::DIM),
        )
            .into_par_iter()
            .for_each(|(s, v)| space.write_vec(s, v));
    }

    #[cfg(feature = "rayon")]
    pub fn par_write_state_costs(&mut self, cost: impl Fn(&S) -> C + Sync)
    where
        C: Send,
        S: Sync,
    {
        use rayon::iter::{
            IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
            ParallelIterator,
        };

        self.states
            .par_iter()
            .zip_eq(self.costs.par_iter_mut())
            .for_each(|(s, c)| *c = cost(s));
    }
}
