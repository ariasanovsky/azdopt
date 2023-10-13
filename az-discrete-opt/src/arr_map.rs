use core::mem::MaybeUninit;

use rayon::prelude::{IntoParallelRefMutIterator, IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::ir_min_tree::{IRState, IRMinTree, Transitions};

// pub const BATCH: usize = 64;
pub const VALUE: usize = 1;

pub fn par_plant_forest<const BATCH: usize, const ACTION: usize, S: IRState + Sync>(
    states: &[S; BATCH],
    predictions: &[[f32; ACTION]; BATCH],
) -> [IRMinTree; BATCH] {
    let mut trees: [MaybeUninit<IRMinTree>; BATCH] = MaybeUninit::uninit_array();
    let states = states.par_iter();
    let predictions = predictions.par_iter();
    trees.par_iter_mut()
        .zip_eq(states.zip_eq(predictions))
        .for_each(|(t, (s, p))| {
            t.write(IRMinTree::new(&s.action_rewards(), p));
        });
    unsafe { MaybeUninit::array_assume_init(trees) }
}

pub fn par_insert_into_forest<const BATCH: usize, const ACTION: usize, S: IRState + Sync> (
    trees: &mut [IRMinTree; BATCH],
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
        .for_each(|(((tree, trans), state), probs)| tree.insert(trans, &state.action_rewards(), probs));
}

pub fn par_forest_observations<const BATCH: usize, const ACTION: usize, const VALUE: usize>
(trees: &[IRMinTree; BATCH]) -> ([[f32; ACTION]; BATCH], [[f32; VALUE]; BATCH]) {
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

pub fn par_update_costs<const BATCH: usize, S: IRState + Sync>(costs: &mut [f32; BATCH], states: &[S; BATCH]) {
    let costs = costs.par_iter_mut();
    let states = states.par_iter();
    costs.zip_eq(states).for_each(|(c, s)| {
        *c = s.cost();
    });
}

pub fn par_simulate_forest_once<const BATCH: usize, S: Send + Sync + Clone + IRState>(
    trees: &[IRMinTree; BATCH],
    roots: &[S; BATCH],
    states: &mut [S; BATCH],
) -> [Transitions; BATCH] {
    let mut transitions: [MaybeUninit<Transitions>; BATCH] = MaybeUninit::uninit_array();
    let trees = trees.par_iter();
    let roots = roots.par_iter();
    let states = states.par_iter_mut();
    trees.zip_eq(roots)
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
    trees: &mut [IRMinTree; BATCH],
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
