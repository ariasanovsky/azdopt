use std::mem::MaybeUninit;

use rayon::iter::{IntoParallelIterator, ParallelIterator};

use crate::{int_min_tree::{INTMinTree, state_data::UpperEstimateData, simulate_once::INTTransitions, transition::{self, INTTransition}, NewTreeLevel}, space::StateActionSpace, path::{ActionPath, ActionPathFor}, state::cost::Cost};

use self::prediction::PredictionData;

pub mod tree;
pub mod state;
pub mod prediction;

pub fn par_simulate_once<'a, P, Space, const BATCH: usize>(
    s_0: &[Space::State; BATCH],
    s_t: &mut [Space::State; BATCH],
    trees: &'a mut [INTMinTree<P>; BATCH],
    p_t: &'a mut [P; BATCH],
    transitions: &mut [Vec<INTTransition<'a>>; BATCH],
    upper_estimate: impl Fn(UpperEstimateData) -> f32 + Sync,
) -> [INTTransitions<'a, P>; BATCH]
where
    Space: StateActionSpace,
    Space::State: Clone + Send,
    P: Send + Sync + ActionPath + ActionPathFor<Space> + Ord,
{
    s_t.clone_from(s_0);
    let mut ends: [_; BATCH] = MaybeUninit::uninit_array();
    (trees, transitions, s_t, p_t, &mut ends)
        .into_par_iter()
        .for_each(|(t, trans, s_t, p_t, end)| {
            p_t.clear();
            trans.clear();
            end.write(t.simulate_once::<Space>(s_t, p_t, trans, &upper_estimate));
        });
    unsafe { MaybeUninit::array_assume_init(ends) }
}

pub fn par_update_existing_nodes<'a, P, Space, const BATCH: usize, const ACTION: usize, const GAIN: usize>(
    nodes: &mut [Option<NewTreeLevel<P>>; BATCH],
    ends: [INTTransitions<'a, P>; BATCH],
    c_t: &[impl Cost<f32> + Sync; BATCH],
    s_t: &[Space::State; BATCH],
    pi_t_theta: &PredictionData<BATCH, ACTION, GAIN>,
    transitions: &mut [Vec<INTTransition<'a>>; BATCH],
) where
    Space: StateActionSpace,
    Space::State: Sync,
    P: ActionPathFor<Space> + Ord + Clone + Send + Sync,
{
    let (g_t_theta, pi_t_theta) = pi_t_theta.get();
    (nodes, ends, c_t, s_t, g_t_theta, pi_t_theta, transitions)
        .into_par_iter()
        .for_each(|(n, end, c_t, s_t, g_t, pi_t, transitions)| {
            *n = end.update_existing_nodes::<Space>(c_t, s_t, pi_t, g_t, transitions);
        });
}

pub fn par_insert_node_at_next_level<'a, P, const BATCH: usize>(
    nodes: &mut [Option<NewTreeLevel<P>>; BATCH],
    trees: &mut [INTMinTree<P>; BATCH],
) where
    P: Ord + Clone + Send,
{
    (nodes, trees)
        .into_par_iter()
        .filter_map(|(n, t)| n.take().map(|n| (n, t)))
        .for_each(|(n, t)| t.insert_node_at_next_level(n));
}