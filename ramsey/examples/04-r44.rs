use core::mem::{MaybeUninit, transmute};

use az_discrete_opt::ir_tree::ir_min_tree::{IRState, ActionsTaken, IRMinTree, Transitions};
use dfdx::optim::Adam;
use dfdx::prelude::{
    cross_entropy_with_logits_loss, mse_loss, DeviceBuildExt, Linear, Module, Optimizer, ReLU,
    ZeroGrads,
};
use dfdx::shapes::Axis;
use dfdx::tensor::{AsArray, AutoDevice, TensorFrom, Trace};
use dfdx::tensor_ops::{AdamConfig, Backward, WeightDecay};
use ramsey::ramsey_state::{ActionVec, GraphState, StateVec, ValueVec, BATCH, STATE, VALUE};
use ramsey::{C, E};
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

const ACTION: usize = C * E;

type Tree = IRMinTree<GraphState>;

// todo! move to ir_min_tree
fn plant_forest(states: &[GraphState; BATCH], predictions: &[ActionVec; BATCH]) -> [Tree; BATCH] {
    let mut trees: [MaybeUninit<Tree>; BATCH] = unsafe {
        let trees: MaybeUninit<[Tree; BATCH]> = MaybeUninit::uninit();
        transmute(trees)
    };
    trees
        .par_iter_mut()
        .zip_eq(states.par_iter().zip_eq(predictions.par_iter()))
        .for_each(|(t, (s, p))| {
            t.write(IRMinTree::new(s, p));
        });
    unsafe { transmute(trees) }
}

struct GraphLogs {
    path: ActionsTaken,
    gain: f32,
}

impl GraphLogs {
    fn empty() -> Self {
        Self {
            path: ActionsTaken::empty(),
            gain: 0.0,
        }
    }

    fn par_new_logs() -> [Self; BATCH] {
        let mut logs: [MaybeUninit<Self>; BATCH] = unsafe {
            let logs: MaybeUninit<[Self; BATCH]> = MaybeUninit::uninit();
            transmute(logs)
        };
        logs.par_iter_mut().for_each(|l| {
            l.write(Self::empty());
        });
        unsafe { transmute(logs) }
    }

    fn update(&mut self, transitions: &Trans) {
        let end = transitions.end();
        let gain = end.gain();
        match gain.total_cmp(&self.gain) {
            std::cmp::Ordering::Greater => {
                self.gain = gain;
                self.path = end.path().clone();
            }
            _ => {}
        }
    }
}

// todo? move to ir_min_tree
fn par_update_logs(logs: &mut [GraphLogs; BATCH], transitions: &[Trans; BATCH]) {
    let logs = logs.par_iter_mut();
    let transitions = transitions.par_iter();
    logs.zip_eq(transitions).for_each(|(l, t)| {
        l.update(t)
    });
}

fn main() {
    const EPOCH: usize = 30;
    const EPISODES: usize = 500;

    // todo!() visuals with https://wandb.ai/site

    // set up model
    let dev = AutoDevice::default();
    let mut core_model = dev.build_module::<Core, f32>();
    let mut logits_model = dev.build_module::<Logits, f32>();
    let mut value_model = dev.build_module::<Valuation, f32>();
    let mut opt = Adam::new(
        &core_model,
        AdamConfig {
            lr: 1e-2,
            betas: [0.5, 0.25],
            eps: 1e-6,
            weight_decay: Some(WeightDecay::Decoupled(1e-2)),
        },
    );

    let mut roots: [GraphState; BATCH] = GraphState::par_generate_batch(5);
    let mut losses: Vec<(f32, f32)> = vec![];

    (1..=EPOCH).for_each(|epoch| {
        println!("==== EPOCH {epoch} ====");
        let mut grads = core_model.alloc_grads();
        let root_vecs: [StateVec; BATCH] = GraphState::par_generate_vecs(&roots);
        // dbg!(root_vecs.get(0).unwrap());

        let root_tensor = dev.tensor(root_vecs.clone());
        let mut prediction_tensor = core_model.forward(root_tensor);
        let mut logits_tensor = logits_model.forward(prediction_tensor.clone());
        let mut probs_tensor = logits_tensor.softmax::<Axis<1>>();
        let mut value_tensor = value_model.forward(prediction_tensor.clone());
        let mut predictions: [ActionVec; BATCH] = probs_tensor.array();
        let mut trees: [Tree; BATCH] = plant_forest(&roots, &predictions);

        let mut logs: [GraphLogs; 64] = GraphLogs::par_new_logs();

        // play episodes
        (1..=EPISODES).for_each(|episode| {
            let (transitions, end_states): ([Trans; BATCH], [GraphState; BATCH]) =
                par_simulate_forest_once(&trees);
            par_update_logs(&mut logs, &transitions);
            if episode % 500 == 0 {
                println!("episode {}", episode);
                trees
                    .chunks_exact(4)
                    .into_iter()
                    .zip(transitions.chunks_exact(4).into_iter())
                    .for_each(|(trees, trans)| {
                        trees.iter().zip(trans.iter()).for_each(|(tree, trans)| {
                            let costs = trans.costs(tree.root_cost());
                            let arr = match &costs[..] {
                                [a, ..] if *a == 0.0 => format!("[{a}] (done)"),
                                c if c.len() <= 10 => format!("{c:?}"),
                                [a, b, c, d, .., w, x, y, z] => {
                                    format!("(l = {}) [{a}, {b}, {c}, {d}, .., {w}, {x}, {y}, {z}]", costs.len())
                                }
                                c => unreachable!("costs = {c:?}")
                            };
                            print!("{arr:64}");
                        });
                        println!();
                    });
                // panic!("done")
            }
            let end_state_vecs: [StateVec; BATCH] = par_state_batch_to_vecs(&end_states);
            prediction_tensor = core_model.forward(dev.tensor(end_state_vecs));
            let logits_tensor = logits_model.forward(prediction_tensor.clone());
            probs_tensor = logits_tensor.clone().softmax::<Axis<1>>();
            value_tensor = value_model.forward(prediction_tensor.clone());
            let probs = probs_tensor.array();
            // let max_deviation_from_one = probs.iter().map(|probs| probs.into_iter().sum::<f32>()).map(|psum| (1.0f32 - psum).abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            // println!("max_deviation_from_one = {}", max_deviation_from_one);
            let values = value_tensor.array();
            par_update_forest(&mut trees, &transitions, &values);
            par_insert_into_forest(&mut trees, &transitions, &end_states, &probs);
        });
        // println!();
        // backprop loss
        let observations: ([ActionVec; BATCH], [ValueVec; BATCH]) = par_forest_observations(&trees);
        let mut entropy = 0.0f32;
        grads = {
            let root_tensor = dev.tensor(root_vecs.clone());
            let traced_predictions = core_model.forward(root_tensor.trace(grads));
            let predicted_logits = logits_model.forward(traced_predictions);
            let observed_probabilities = dev.tensor(observations.0.clone());
            let cross_entropy =
                cross_entropy_with_logits_loss(predicted_logits, observed_probabilities);
            entropy = cross_entropy.array();
            cross_entropy.backward()
        };
        grads = {
            let root_tensor = dev.tensor(root_vecs.clone());
            let traced_predictions = core_model.forward(root_tensor.trace(grads));
            let predicted_values = value_model.forward(traced_predictions);
            let observed_values = dev.tensor(observations.1.clone());
            let square_error = mse_loss(predicted_values, observed_values);
            losses.push((entropy, square_error.array()));
            println!("{losses:?}");
            square_error.backward()
        };
        opt.update(&mut core_model, &grads).unwrap();
        core_model.zero_grads(&mut grads);

        par_update_roots(&mut roots, &logs, 5 * epoch);
    });
}

// todo! move to ir_min_tree
fn par_update_roots(roots: &mut [GraphState; BATCH], logs: &[GraphLogs; BATCH], time: usize) {
    let roots = roots.par_iter_mut();
    let logs = logs.par_iter();
    roots.zip_eq(logs).for_each(|(root, log)| {
        root.apply(&log.path);
        root.reset(time);
    });
}

// todo! move to ir_min_tree
fn par_insert_into_forest(
    trees: &mut [Tree; BATCH],
    transitions: &[Trans; BATCH],
    end_states: &[GraphState; BATCH],
    probs: &[ActionVec; BATCH],
) {
    let trees = trees.par_iter_mut();
    let trans = transitions.par_iter();
    let end_states = end_states.par_iter();
    let probs = probs.par_iter();
    trees
        .zip_eq(trans)
        .zip_eq(end_states)
        .zip_eq(probs)
        .for_each(|(((tree, trans), state), probs)| tree.insert(trans, state, probs));
}

// todo! move to ir_min_tree
fn par_forest_observations(trees: &[Tree; BATCH]) -> ([ActionVec; BATCH], [ValueVec; BATCH]) {
    let mut probabilities: [ActionVec; BATCH] = [[0.0f32; ACTION]; BATCH];
    let mut values: [ValueVec; BATCH] = [[0.0f32; VALUE]; BATCH];
    trees
        .par_iter()
        .zip_eq(probabilities.par_iter_mut())
        .zip_eq(values.par_iter_mut())
        .for_each(|((tree, p), v)| {
            let observations = tree.observations();
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

// todo!() update values separately from inserting new path
fn par_update_forest(
    trees: &mut [Tree; BATCH],
    transitions: &[Trans; BATCH],
    values: &[ValueVec; BATCH],
) {
    trees
        .par_iter_mut()
        .zip_eq(transitions.par_iter())
        .zip_eq(values.par_iter())
        .for_each(|((tree, trans), values)| {
            tree.update(trans, values);
        });
}

// todo! move to ir_min_tree
fn par_state_batch_to_vecs(states: &[GraphState; BATCH]) -> [StateVec; BATCH] {
    let mut state_vecs: [MaybeUninit<StateVec>; BATCH] = unsafe {
        let state_vecs: MaybeUninit<[StateVec; BATCH]> = MaybeUninit::uninit();
        transmute(state_vecs)
    };
    state_vecs
        .par_iter_mut()
        .zip_eq(states.par_iter())
        .for_each(|(v, s)| {
            v.write(s.to_vec());
        });
    unsafe { transmute(state_vecs) }
}

// todo! move to ir_min_tree
fn par_simulate_forest_once(trees: &[Tree; BATCH]) -> ([Trans; BATCH], [GraphState; BATCH]) {
    let mut transitions: [MaybeUninit<Trans>; BATCH] = unsafe {
        let transitions: MaybeUninit<[Trans; BATCH]> = MaybeUninit::uninit();
        transmute(transitions)
    };
    let mut state_vecs: [MaybeUninit<GraphState>; BATCH] = unsafe {
        let state_vecs: MaybeUninit<[GraphState; BATCH]> = MaybeUninit::uninit();
        transmute(state_vecs)
    };
    trees
        .par_iter()
        .zip_eq(transitions.par_iter_mut())
        .zip_eq(state_vecs.par_iter_mut())
        .for_each(|((tree, t), s)| {
            let (trans, state) = tree.simulate_once();
            t.write(trans);
            s.write(state);
        });
    unsafe { (transmute(transitions), transmute(state_vecs)) }
}

type Trans = Transitions;

const HIDDEN_1: usize = 256;
const HIDDEN_2: usize = 128;

type Core = (
    (Linear<STATE, HIDDEN_1>, ReLU),
    (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
    // Linear<HIDDEN_2, PREDICTION>,
);

type Logits = (
    Linear<HIDDEN_2, ACTION>,
    // Softmax,
);

type Valuation = (
    Linear<HIDDEN_2, 1>,
    // Linear<HIDDEN_2, 3>,
);
