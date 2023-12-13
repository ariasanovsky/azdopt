use crate::{
    int_min_tree::{
        simulate_once::EndNodeAndLevel, state_data::UpperEstimateData, transition::INTTransition,
        INTMinTree, NewTreeLevel,
    },
    path::ActionPathFor,
    space::StateActionSpace,
    state::cost::Cost,
};

use super::{prediction::PredictionData, state::StateData};

type Dummy = [u64; 3];
const _VALID: () = SameSizeAndAlignment::<INTTransition<'_>, Dummy>::SAME_SIZE_AND_ALIGNMENT;

pub struct TreeData<P> {
    trees: Vec<INTMinTree<P>>,
    paths: Vec<P>,
    nodes: Vec<Option<NewTreeLevel<P>>>,
    transitions: Vec<Vec<Dummy>>,
}

struct SameSizeAndAlignment<T, U>(core::marker::PhantomData<(T, U)>);

impl<T, U> SameSizeAndAlignment<T, U> {
    const SAME_SIZE_AND_ALIGNMENT: () = {
        assert!(core::mem::size_of::<T>() == core::mem::size_of::<U>());
        assert!(core::mem::align_of::<T>() == core::mem::align_of::<U>());
    };
}

#[allow(clippy::let_unit_value)]
pub fn reuse_as<T, U>(t: &mut Vec<T>) -> &mut Vec<U> {
    // there's a prettier solution locked behind https://github.com/rust-lang/rust/issues/76001
    let _: () = SameSizeAndAlignment::<T, U>::SAME_SIZE_AND_ALIGNMENT; // this should be `const`
    t.clear();
    unsafe { core::mem::transmute(t) }
}

pub struct Ends<'new_nodes, P> {
    ends: Vec<EndNodeAndLevel<'new_nodes, P>>,
    paths: &'new_nodes [P],
    nodes: &'new_nodes mut [Option<NewTreeLevel<P>>],
    transitions: &'new_nodes mut [Vec<INTTransition<'new_nodes>>],
}

impl<'new_nodes, P> Ends<'new_nodes, P> {
    #[cfg(feature = "rayon")]
    pub fn par_update_existing_nodes<'a, Space>(
        self,
        space: &Space,
        c_t: &[impl Cost<f32> + Sync],
        s_t: &[Space::State],
        predictions: &PredictionData<'a>,
        action: usize,
        gain: usize,
    ) where
        Space: StateActionSpace + Sync,
        Space::State: Sync,
        P: ActionPathFor<Space> + Ord + Clone + Send + Sync,
    {
        use rayon::{slice::ParallelSlice, iter::{IntoParallelIterator, ParallelIterator}};

        let Self {
            ends,
            paths,
            nodes,
            transitions,
        } = self;
        let (pi_t_theta, g_t_theta) = predictions.get();
        (
            ends,
            paths,
            nodes,
            c_t,
            s_t,
            g_t_theta.par_chunks_exact(gain),
            pi_t_theta.par_chunks_exact(action),
            transitions,
        )
            .into_par_iter()
            .for_each(|(end, p_t, n, c_t, s_t, g_t, pi_t, transitions)| {
                *n = end.update_existing_nodes(space, c_t, s_t, p_t, pi_t, g_t, transitions);
            });
    }
}

impl<P> TreeData<P> {
    #[cfg(feature = "rayon")]
    pub fn par_new<'a, Space>(
        space: &Space,
        add_noise: impl Fn(usize, &mut [f32]) + Sync,
        predictions: &mut PredictionData<'a>,
        states: &StateData<'a, Space::State, impl Cost<f32> + Sync>,
        batch: usize,
        action: usize,
    ) -> Self
    where
        P: Clone + Ord + Send + Sync + ActionPathFor<Space>,
        Space: StateActionSpace + Sync,
        Space::State: Send + Sync + Clone,
    {
        use rayon::{slice::ParallelSliceMut, iter::{IntoParallelIterator, IndexedParallelIterator, ParallelIterator}};

        let trees =
        (
            predictions.pi_mut().par_chunks_exact_mut(action),
            states.get_costs(),
            states.get_states(),
        )
            .into_par_iter()
            .enumerate()
            .map(|(i, (pi_0_theta, c_0, s_0))| {
                add_noise(i, pi_0_theta);
                INTMinTree::new(space, pi_0_theta, c_0.evaluate(), s_0)
            }).collect::<Vec<_>>();
        let paths = (0..batch).map(|_| P::new()).collect::<Vec<_>>();
        let nodes = (0..batch).map(|_| None).collect::<Vec<_>>();
        let transitions = (0..batch).map(|_| Vec::new()).collect::<Vec<_>>();
        Self {
            trees,
            paths,
            nodes,
            transitions,
        }
    }

    // pub fn new(
    //     trees: [INTMinTree<P>; BATCH],
    //     paths: [P; BATCH],
    //     nodes: [Option<NewTreeLevel<P>>; BATCH],
    // ) -> Self {
    //     let transitions = core::array::from_fn(|_| Vec::new());
    //     Self {
    //         trees,
    //         paths,
    //         nodes,
    //         transitions,
    //     }
    // }

    pub fn trees_mut(&mut self) -> &mut [INTMinTree<P>] {
        &mut self.trees
    }

    pub fn trees(&self) -> &[INTMinTree<P>] {
        &self.trees
    }

    #[cfg(feature = "rayon")]
    pub fn par_simulate_once<Space>(
        &mut self,
        space: &Space,
        s_t: &mut [Space::State],
        upper_estimate: impl Fn(UpperEstimateData) -> f32 + Sync,
    ) -> Ends<'_, P>
    where
        Space: StateActionSpace + Sync,
        Space::State: Send,
        P: Send + Sync + ActionPathFor<Space> + Ord,
    {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        let Self {
            trees,
            paths,
            nodes,
            transitions,
        } = self;
        let ends =
        (trees, transitions, s_t, paths)
            .into_par_iter()
            .map(|(t, trans, s_t, p_t)| {
                p_t.clear();
                t.simulate_once(space, s_t, p_t, reuse_as(trans), &upper_estimate)
            }).collect::<Vec<_>>();
        Ends {
            ends,
            paths: &self.paths,
            nodes,
            transitions: unsafe { core::mem::transmute(&mut self.transitions[..]) },
        }
    }

    #[cfg(feature = "rayon")]
    pub fn par_insert_nodes(&mut self)
    where
        P: Clone + Ord + Send,
    {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        let Self {
            trees,
            paths: _,
            nodes,
            transitions: _,
        } = self;
        (nodes, trees)
            .into_par_iter()
            .filter_map(|(n, t)| n.take().map(|n| (n, t)))
            .for_each(|(n, t)| t.insert_node_at_next_level(n));
    }
}
