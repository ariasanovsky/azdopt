use core::mem::MaybeUninit;

use rayon::iter::{IntoParallelIterator, IndexedParallelIterator, ParallelIterator};

use crate::space::StateActionSpace;

pub struct StateData<const BATCH: usize, const STATE: usize, S, C> {
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
        let mut states: [MaybeUninit<S>; BATCH] = MaybeUninit::uninit_array();
        let mut costs: [MaybeUninit<C>; BATCH] = MaybeUninit::uninit_array();
        let mut vectors: [[f32; STATE]; BATCH] = [[0.0; STATE]; BATCH];
        (&mut states, &mut costs, &mut vectors)
            .into_par_iter()
            .enumerate()
            .for_each(|(i, (s, c, v))| {
                let mut rng = rng(i);
                s.write(state(i, &mut rng));
                c.write(cost(unsafe { s.assume_init_ref() }));
                Space::write_vec(unsafe { s.assume_init_ref() }, v);
            });
        let states = unsafe { MaybeUninit::array_assume_init(states) };
        let costs = unsafe { MaybeUninit::array_assume_init(costs) };
        Self {
            states,
            costs,
            vectors,
        }
    }
}