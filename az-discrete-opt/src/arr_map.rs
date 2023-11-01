use core::mem::MaybeUninit;

use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{
    iq_min_tree::{IQMinTree, IQState, Transitions},
    log::BasicLog,
};

// pub const BATCH: usize = 64;
pub const VALUE: usize = 1;

pub fn par_plant_forest<
    const BATCH: usize,
    const ACTION: usize,
    const STATE: usize,
    S: IQState<STATE> + Sync,
>(
    states: &[S; BATCH],
    predictions: &[[f32; ACTION]; BATCH],
) -> [IQMinTree; BATCH] {
    let mut trees: [MaybeUninit<IQMinTree>; BATCH] = MaybeUninit::uninit_array();
    let states = states.par_iter();
    let predictions = predictions.par_iter();
    trees
        .par_iter_mut()
        .zip_eq(states.zip_eq(predictions))
        .for_each(|(t, (s, p))| {
            t.write(IQMinTree::new(&s.action_rewards(), p));
        });
    unsafe { MaybeUninit::array_assume_init(trees) }
}

pub fn par_insert_into_forest<
    const BATCH: usize,
    const ACTION: usize,
    const STATE: usize,
    S: IQState<STATE> + Sync,
>(
    trees: &mut [IQMinTree; BATCH],
    transitions: &[Transitions; BATCH],
    end_states: &[S; BATCH],
    probs: &[[f32; ACTION]; BATCH],
) {
    let trees = trees.par_iter_mut();
    let trans = transitions.par_iter();
    let end_states = end_states.par_iter();
    let probs = probs.par_iter();
    trees
        .zip_eq(trans)
        .zip_eq(end_states)
        .zip_eq(probs)
        .for_each(|(((tree, trans), state), probs)| {
            tree.insert(trans, &state.action_rewards(), probs)
        });
}

pub fn par_forest_observations<const BATCH: usize, const ACTION: usize, const VALUE: usize>(
    trees: &[IQMinTree; BATCH],
) -> ([[f32; ACTION]; BATCH], [[f32; VALUE]; BATCH]) {
    let mut probabilities: [_; BATCH] = [[0.0f32; ACTION]; BATCH];
    let mut values: [_; BATCH] = [[0.0f32; VALUE]; BATCH];
    trees
        .par_iter()
        .zip_eq(probabilities.par_iter_mut())
        .zip_eq(values.par_iter_mut())
        .for_each(|((tree, p), v)| {
            let observations = tree.observations::<ACTION>();
            let (prob, value) = observations.split_at(ACTION);
            p.iter_mut().zip(prob.iter()).for_each(|(p, prob)| {
                *p = *prob;
            });
            v.iter_mut().zip(value.iter()).for_each(|(v, value)| {
                *v = *value;
            });
        });
    (probabilities, values)
}

pub fn par_set_costs<const BATCH: usize, S: Sync>(
    costs: &mut [f32; BATCH],
    states: &[S; BATCH],
    cost: &(impl Fn(&S) -> f32 + Sync),
) {
    let costs = costs.par_iter_mut();
    let states = states.par_iter();
    costs.zip_eq(states).for_each(|(c, s)| {
        *c = cost(s);
    });
}

pub fn par_simulate_forest_once<
    const BATCH: usize,
    const STATE: usize,
    S: IQState<STATE> + Send + Sync + Clone,
>(
    trees: &[IQMinTree; BATCH],
    roots: &[S; BATCH],
    states: &mut [S; BATCH],
) -> [Transitions; BATCH] {
    let mut transitions: [MaybeUninit<Transitions>; BATCH] = MaybeUninit::uninit_array();
    let trees = trees.par_iter();
    let roots = roots.par_iter();
    let states = states.par_iter_mut();
    trees
        .zip_eq(roots)
        .zip_eq(transitions.par_iter_mut())
        .zip_eq(states)
        .for_each(|(((tree, root), t), state)| {
            *state = root.clone();
            let trans = tree.simulate_once(state);
            t.write(trans);
        });
    unsafe { MaybeUninit::array_assume_init(transitions) }
}

pub fn par_update_forest<const BATCH: usize>(
    trees: &mut [IQMinTree; BATCH],
    transitions: &[Transitions; BATCH],
    values: &[[f32; VALUE]; BATCH],
) {
    trees
        .par_iter_mut()
        .zip_eq(transitions.par_iter())
        .zip_eq(values.par_iter())
        .for_each(|((tree, trans), values)| {
            tree.update(trans, values);
        });
}

pub fn par_use_logged_roots<const BATCH: usize, const STATE: usize, S: IQState<STATE> + Send>(
    roots: &mut [S; BATCH],
    logs: &mut [BasicLog; BATCH],
    time: usize,
) {
    let roots = roots.par_iter_mut();
    let logs = logs.par_iter_mut();
    roots.zip_eq(logs).for_each(|(root, log)| {
        root.apply(&log.path());
        root.reset(time);
    });
}

pub fn par_state_batch_to_vecs<const BATCH: usize, const STATE: usize, S: IQState<STATE> + Sync>(
    states: &[S; BATCH],
) -> [[f32; STATE]; BATCH] {
    let mut state_vecs: [MaybeUninit<[f32; STATE]>; BATCH] = MaybeUninit::uninit_array();
    state_vecs
        .par_iter_mut()
        .zip_eq(states.par_iter())
        .for_each(|(v, s)| {
            v.write(s.to_vec());
        });
    unsafe { MaybeUninit::array_assume_init(state_vecs) }
}
