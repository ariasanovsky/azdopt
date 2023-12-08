use core::mem::MaybeUninit;

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

use crate::space::StateActionSpace;

pub struct StateData<const BATCH: usize, const STATE: usize, S, C> {
    roots: [S; BATCH],
    states: [S; BATCH],
    costs: [C; BATCH],
    vectors: [[f32; STATE]; BATCH],
}

impl<const BATCH: usize, const STATE: usize, S, C> StateData<BATCH, STATE, S, C> {
    pub fn par_new<Space, R>(
        rng: impl Fn(usize) -> R + Sync,
        state: impl Fn(usize, &mut R) -> S + Sync,
        cost: impl Fn(&S) -> C + Sync,
    ) -> Self
    where
        Space: StateActionSpace<State = S>,
        S: Send + Sync + Clone,
        C: Send + Sync,
    {
        let mut roots: [MaybeUninit<S>; BATCH] = MaybeUninit::uninit_array();
        let mut costs: [MaybeUninit<C>; BATCH] = MaybeUninit::uninit_array();
        let mut vectors: [[f32; STATE]; BATCH] = [[0.0; STATE]; BATCH];
        (&mut roots, &mut costs, &mut vectors)
            .into_par_iter()
            .enumerate()
            .for_each(|(i, (r, c, v))| {
                let mut rng = rng(i);
                r.write(state(i, &mut rng));
                c.write(cost(unsafe { r.assume_init_ref() }));
                Space::write_vec(unsafe { r.assume_init_ref() }, v);
            });
        let roots = unsafe { MaybeUninit::array_assume_init(roots) };
        let states = roots.clone();
        let costs = unsafe { MaybeUninit::array_assume_init(costs) };
        Self {
            roots,
            states,
            costs,
            vectors,
        }
    }

    pub fn get_states(&self) -> &[S; BATCH] {
        &self.states
    }

    pub fn get_states_mut(&mut self) -> &mut [S; BATCH] {
        &mut self.states
    }

    pub fn get_costs(&self) -> &[C; BATCH] {
        &self.costs
    }

    pub fn get_vectors(&self) -> &[[f32; STATE]; BATCH] {
        &self.vectors
    }

    pub fn reset_states(&mut self)
    where
        S: Clone,
    {
        self.states.clone_from(&self.roots);
    }

    pub fn get_roots_mut(&mut self) -> &mut [S; BATCH] {
        &mut self.roots
    }

    pub fn par_write_state_vecs<Space>(&mut self)
    where
        Space: StateActionSpace<State = S>,
        S: Sync,
    {
        (&self.states, &mut self.vectors)
            .into_par_iter()
            .for_each(|(s, v)| Space::write_vec(s, v));
    }

    pub fn par_write_state_costs(&mut self, cost: impl Fn(&S) -> C + Sync)
    where
        C: Send,
        S: Sync,
    {
        (&self.states, &mut self.costs)
            .into_par_iter()
            .for_each(|(s, c)| *c = cost(s));
    }
}
