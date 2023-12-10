use std::mem::MaybeUninit;

use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

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

pub struct TreeData<const BATCH: usize, P> {
    trees: [INTMinTree<P>; BATCH],
    paths: [P; BATCH],
    nodes: [Option<NewTreeLevel<P>>; BATCH],
    transitions: [Vec<Dummy>; BATCH],
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

pub struct Ends<'a, const BATCH: usize, P> {
    ends: [EndNodeAndLevel<'a, P>; BATCH],
    paths: &'a [P; BATCH],
    nodes: &'a mut [Option<NewTreeLevel<P>>; BATCH],
    transitions: &'a mut [Vec<INTTransition<'a>>; BATCH],
}

impl<'a, const BATCH: usize, P> Ends<'a, BATCH, P> {
    #[cfg(feature = "rayon")]
    pub fn par_update_existing_nodes<Space, const ACTION: usize, const GAIN: usize>(
        self,
        c_t: &[impl Cost<f32> + Sync; BATCH],
        s_t: &[Space::State; BATCH],
        predictions: &PredictionData<BATCH, ACTION, GAIN>,
    ) where
        Space: StateActionSpace,
        Space::State: Sync,
        P: ActionPathFor<Space> + Ord + Clone + Send + Sync,
    {
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
            g_t_theta,
            pi_t_theta,
            transitions,
        )
            .into_par_iter()
            .for_each(|(end, p_t, n, c_t, s_t, g_t, pi_t, transitions)| {
                *n = end.update_existing_nodes::<Space>(c_t, s_t, p_t, pi_t, g_t, transitions);
            });
    }
}

impl<const BATCH: usize, P> TreeData<BATCH, P> {
    pub fn par_new<const STATE: usize, const ACTION: usize, const GAIN: usize, Space, C>(
        add_noise: impl Fn(usize, &mut [f32]) + Sync,
        predictions: &mut PredictionData<BATCH, ACTION, GAIN>,
        states: &StateData<BATCH, STATE, Space::State, C>,
    ) -> Self
    where
        P: Clone + Ord + Send + Sync + ActionPathFor<Space>,
        Space: StateActionSpace,
        C: Cost<f32> + Sync,
        Space::State: Send + Sync + Clone,
    {
        let mut trees: [MaybeUninit<INTMinTree<P>>; BATCH] = MaybeUninit::uninit_array();
        (
            &mut trees,
            predictions.pi_mut(),
            states.get_costs(),
            states.get_states(),
        )
            .into_par_iter()
            .enumerate()
            .for_each(|(i, (t, pi_0_theta, c_0, s_0))| {
                add_noise(i, pi_0_theta);
                t.write(INTMinTree::new::<Space>(pi_0_theta, c_0.evaluate(), s_0));
            });
        let trees = unsafe { MaybeUninit::array_assume_init(trees) };
        let mut paths: [MaybeUninit<P>; BATCH] = MaybeUninit::uninit_array();
        let mut nodes: [MaybeUninit<Option<NewTreeLevel<P>>>; BATCH] = MaybeUninit::uninit_array();
        let mut transitions: [MaybeUninit<Vec<Dummy>>; BATCH] = MaybeUninit::uninit_array();
        (&mut paths, &mut nodes, &mut transitions)
            .into_par_iter()
            .for_each(|(p, n, trans)| {
                p.write(P::new());
                n.write(None);
                trans.write(Vec::new());
            });
        let paths = unsafe { MaybeUninit::array_assume_init(paths) };
        let nodes = unsafe { MaybeUninit::array_assume_init(nodes) };
        let transitions = unsafe { MaybeUninit::array_assume_init(transitions) };
        Self {
            trees,
            paths,
            nodes,
            transitions,
        }
    }

    pub fn new(
        trees: [INTMinTree<P>; BATCH],
        paths: [P; BATCH],
        nodes: [Option<NewTreeLevel<P>>; BATCH],
    ) -> Self {
        let transitions = core::array::from_fn(|_| Vec::new());
        Self {
            trees,
            paths,
            nodes,
            transitions,
        }
    }

    pub fn trees_mut(&mut self) -> &mut [INTMinTree<P>; BATCH] {
        &mut self.trees
    }

    pub fn trees(&self) -> &[INTMinTree<P>; BATCH] {
        &self.trees
    }

    #[cfg(feature = "rayon")]
    pub fn par_simulate_once<Space>(
        &mut self,
        s_t: &mut [Space::State; BATCH],
        upper_estimate: impl Fn(UpperEstimateData) -> f32 + Sync,
    ) -> Ends<'_, BATCH, P>
    where
        Space: StateActionSpace,
        Space::State: Send,
        P: Send + Sync + ActionPathFor<Space> + Ord,
    {
        let Self {
            trees,
            paths,
            nodes,
            transitions,
        } = self;
        let mut ends: [_; BATCH] = MaybeUninit::uninit_array();
        (trees, transitions, s_t, paths, &mut ends)
            .into_par_iter()
            .for_each(|(t, trans, s_t, p_t, end)| {
                p_t.clear();
                end.write(t.simulate_once::<Space>(s_t, p_t, reuse_as(trans), &upper_estimate));
            });
        let ends = unsafe { MaybeUninit::array_assume_init(ends) };
        Ends {
            ends,
            paths: &self.paths,
            nodes,
            transitions: unsafe { core::mem::transmute(&mut self.transitions) },
        }
    }

    #[cfg(feature = "rayon")]
    pub fn par_insert_nodes(&mut self)
    where
        P: Clone + Ord + Send,
    {
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
