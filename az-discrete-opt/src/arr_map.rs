use core::mem::{MaybeUninit, transmute};

use rayon::prelude::{IntoParallelRefMutIterator, IndexedParallelIterator, IntoParallelRefIterator, ParallelIterator};

use crate::ir_min_tree::{IRState, IRMinTree, Transitions};

pub const BATCH: usize = 64;

pub fn par_plant_forest<const ACTION: usize, S: IRState + Sync>(
    states: &[S; BATCH],
    predictions: &[[f32; ACTION]; BATCH],
) -> [IRMinTree; BATCH] {
    let mut trees: [MaybeUninit<IRMinTree>; BATCH] = unsafe {
        let trees: MaybeUninit<[IRMinTree; BATCH]> = MaybeUninit::uninit();
        transmute(trees)
    };
    let states = states.par_iter();
    let predictions = predictions.par_iter();
    trees.par_iter_mut()
        .zip_eq(states.zip_eq(predictions))
        .for_each(|(t, (s, p))| {
            t.write(IRMinTree::new(&s.action_rewards(), p));
        });
    unsafe { transmute(trees) }
}

pub fn par_insert_into_forest<const ACTION: usize, S: IRState + Sync> (
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

pub fn par_forest_observations<const ACTION: usize, const VALUE: usize>
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

// fn par_state_batch_to_vecs<const STATE: usize, S: IRState + Sync>(states: &[S; BATCH]) -> [[f32; STATE]; BATCH] {
//     let mut state_vecs: [MaybeUninit<[f32; STATE]>; BATCH] = unsafe {
//         let state_vecs: MaybeUninit<[[f32; STATE]; BATCH]> = MaybeUninit::uninit();
//         transmute(state_vecs)
//     };
//     state_vecs
//         .par_iter_mut()
//         .zip_eq(states.par_iter())
//         .for_each(|(v, s)| {
//             v.write(s.to_vec());
//         });
//     unsafe { transmute(state_vecs) }
// }

// todo! move to ir_min_tree
// fn par_simulate_forest_once<S: IRState + Send + Sync + Clone>(trees: &[IRMinTree; BATCH], roots: &[S; BATCH]) -> ([Transitions; BATCH], [S; BATCH]) {
//     let mut transitions: [MaybeUninit<Transitions>; BATCH] = unsafe {
//         let transitions: MaybeUninit<[Transitions; BATCH]> = MaybeUninit::uninit();
//         transmute(transitions)
//     };
//     let mut state_vecs: [MaybeUninit<S>; BATCH] = unsafe {
//         let state_vecs: MaybeUninit<[S; BATCH]> = MaybeUninit::uninit();
//         transmute(state_vecs)
//     };
//     let trees = trees.par_iter();
//     let roots = roots.par_iter();
//     trees.zip_eq(roots)
//         .zip_eq(transitions.par_iter_mut())
//         .zip_eq(state_vecs.par_iter_mut())
//         .for_each(|(((tree, root), t), s)| {
//             let (trans, state) = tree.simulate_once(root);
//             t.write(trans);
//             s.write(state);
//         });
//     unsafe { (transmute(transitions), transmute(state_vecs)) }
// }
