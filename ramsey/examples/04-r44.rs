use core::mem::MaybeUninit;
use core::mem::transmute;

use az_discrete_opt::ir_tree::ir_min_tree::IRMinTree;
use az_discrete_opt::ir_tree::ir_min_tree::IRState;
use az_discrete_opt::ir_tree::ir_min_tree::Transitions;
use bit_iter::BitIter;
use dfdx::optim::Adam;
use dfdx::prelude::Optimizer;
use dfdx::prelude::ZeroGrads;
use dfdx::prelude::cross_entropy_with_logits_loss;
use dfdx::prelude::mse_loss;
use dfdx::prelude::{Linear, ReLU, DeviceBuildExt, Module};
use dfdx::shapes::Axis;
use dfdx::tensor::Trace;
use dfdx::tensor::{AutoDevice, TensorFrom, AsArray};
use dfdx::tensor_ops::Backward;
use dfdx::tensor_ops::{AdamConfig, WeightDecay};
use itertools::Itertools;
use ramsey::ramsey_state::ActionVec;
use ramsey::ramsey_state::BATCH;
use ramsey::ramsey_state::GraphState;
use ramsey::ramsey_state::STATE;
use ramsey::ramsey_state::StateVec;
use ramsey::ramsey_state::VALUE;
use ramsey::ramsey_state::ValueVec;
use ramsey::ramsey_state::edge_from_position;
use ramsey::ramsey_state::edge_to_position;
use ramsey::{ColoredCompleteGraph, MulticoloredGraphEdges, MulticoloredGraphNeighborhoods, OrderedEdgeRecolorings, CliqueCounts, C, E, Color, N, EdgeRecoloring};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator};

const ACTION: usize = C * E;

type Tree = IRMinTree<GraphState>;

fn plant_forest(states: &[GraphState; BATCH], predictions: &[ActionVec; BATCH]) -> [Tree; BATCH] {
    let mut trees: [MaybeUninit<Tree>; BATCH] = unsafe {
        let trees: MaybeUninit<[Tree; BATCH]> = MaybeUninit::uninit();
        transmute(trees)
    };
    trees.par_iter_mut().zip_eq(states.par_iter().zip_eq(predictions.par_iter())).for_each(|(t, (s, p))| {
        t.write(IRMinTree::new(s, p));
    });
    unsafe {
        transmute(trees)
    }
}

fn main() {
    const EPOCH: usize = 30;
    const EPISODES: usize = 100;

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
         }
    );
    
    (0..EPOCH).for_each(|epoch| {
        println!("==== EPOCH {epoch} ====");
        let mut grads = core_model.alloc_grads();
        let roots: [GraphState; BATCH] = GraphState::par_generate_batch(20);
        let root_vecs: [StateVec; BATCH] = GraphState::par_generate_vecs(&roots);
        // dbg!(root_vecs.get(0).unwrap());
        
        let root_tensor = dev.tensor(root_vecs.clone());
        let mut prediction_tensor = core_model.forward(root_tensor);
        let mut logits_tensor = logits_model.forward(prediction_tensor.clone());
        let mut probs_tensor = logits_tensor.softmax::<Axis<1>>();
        let mut value_tensor = value_model.forward(prediction_tensor.clone());
        let mut predictions: [ActionVec; BATCH] = probs_tensor.array();
        let mut trees: [Tree; BATCH] = plant_forest(&roots, &predictions);
        
        // let mut longest_paths: Vec<usize> = vec![];

        // play episodes
        (0..EPISODES).for_each(|episode| {
            let (transitions, end_states): ([Trans; BATCH], [GraphState; BATCH]) = simulate_forest_once(&trees);
            // let longest_path = transitions.iter().map(|t| t.len()).max().unwrap();
            // longest_paths.push(longest_path);
            if EPISODES % 20 == 0 {
                trees.iter().zip(transitions.iter()).for_each(|(tree, trans)| {
                    println!("{:?}", trans.costs(tree.root_cost()));
                });
            }
            let end_state_vecs: [StateVec; BATCH] = state_batch_to_vecs(&end_states);
            prediction_tensor = core_model.forward(dev.tensor(end_state_vecs));
            let logits_tensor = logits_model.forward(prediction_tensor.clone());
            probs_tensor = logits_tensor.clone().softmax::<Axis<1>>();
            value_tensor = value_model.forward(prediction_tensor.clone());
            let probs = probs_tensor.array();
            // let max_deviation_from_one = probs.iter().map(|probs| probs.into_iter().sum::<f32>()).map(|psum| (1.0f32 - psum).abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            // println!("max_deviation_from_one = {}", max_deviation_from_one);
            let values = value_tensor.array();
            update_forest(&mut trees, &transitions, &values);
            insert_into_forest(&mut trees, &transitions, &end_states, &probs);
        });
        // println!();
        // backprop loss
        let observations: ([ActionVec; BATCH], [ValueVec; BATCH]) = forest_observations(&trees);
        grads = {
            let root_tensor = dev.tensor(root_vecs.clone());
            let traced_predictions = core_model.forward(root_tensor.trace(grads));
            let predicted_logits = logits_model.forward(traced_predictions);
            let observed_probabilities = dev.tensor(observations.0.clone());
            let cross_entropy = cross_entropy_with_logits_loss(predicted_logits, observed_probabilities);
            print!("{:10}\t", cross_entropy.array());
            cross_entropy.backward()
        };
        grads = {
            let root_tensor = dev.tensor(root_vecs.clone());
            let traced_predictions = core_model.forward(root_tensor.trace(grads));
            let predicted_values = value_model.forward(traced_predictions);
            let observed_values = dev.tensor(observations.1.clone());
            let square_error = mse_loss(predicted_values, observed_values);
            println!("{}", square_error.array());
            square_error.backward()
        };
        opt.update(&mut core_model, &grads).unwrap();
        core_model.zero_grads(&mut grads);
    });
}

fn insert_into_forest(
    trees: &mut [Tree; BATCH],
    transitions: &[Trans; BATCH],
    end_states: &[GraphState; BATCH],
    probs: &[ActionVec; BATCH]
) {
    let trees = trees.par_iter_mut();
    let trans = transitions.par_iter();
    let end_states = end_states.par_iter();
    let probs = probs.par_iter();
    trees.zip_eq(trans).zip_eq(end_states).zip_eq(probs).for_each(|(((tree, trans), state), probs)| {
        tree.insert(trans, state, probs)
    });
}

fn forest_observations(trees: &[Tree; BATCH]) -> ([ActionVec; BATCH], [ValueVec; BATCH]) {
    let mut probabilities: [ActionVec; BATCH] = [[0.0f32; ACTION]; BATCH];
    let mut values: [ValueVec; BATCH] = [[0.0f32; VALUE]; BATCH];
    trees.par_iter().zip_eq(probabilities.par_iter_mut()).zip_eq(values.par_iter_mut()).for_each(|((tree, p), v)| {
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
fn update_forest(
    trees: &mut [Tree; BATCH],
    transitions: &[Trans; BATCH],
    values: &[ValueVec; BATCH],
) {
    trees.par_iter_mut()
        .zip_eq(transitions.par_iter())
        .zip_eq(values.par_iter())
        .for_each(|((tree, trans), values)| {
        tree.update(trans,values);
    });
}

fn state_batch_to_vecs(states: &[GraphState; BATCH]) -> [StateVec; BATCH] {
    let mut state_vecs: [MaybeUninit<StateVec>; BATCH] = unsafe {
        let state_vecs: MaybeUninit<[StateVec; BATCH]> = MaybeUninit::uninit();
        transmute(state_vecs)
    };
    state_vecs.par_iter_mut().zip_eq(states.par_iter()).for_each(|(v, s)| {
        v.write(s.to_vec());
    });
    unsafe {
        transmute(state_vecs)
    }
}

fn simulate_forest_once(trees: &[Tree; BATCH]) -> ([Trans; BATCH], [GraphState; BATCH]) {
    let mut transitions: [MaybeUninit<Trans>; BATCH] = unsafe {
        let transitions: MaybeUninit<[Trans; BATCH]> = MaybeUninit::uninit();
        transmute(transitions)
    };
    let mut state_vecs: [MaybeUninit<GraphState>; BATCH] = unsafe {
        let state_vecs: MaybeUninit<[GraphState; BATCH]> = MaybeUninit::uninit();
        transmute(state_vecs)
    };
    trees.par_iter().zip_eq(transitions.par_iter_mut()).zip_eq(state_vecs.par_iter_mut()).for_each(|((tree, t), s)| {
        let (trans, state) = tree.simulate_once();
        t.write(trans);
        s.write(state);
    });
    unsafe {
        (transmute(transitions), transmute(state_vecs))
    }
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
