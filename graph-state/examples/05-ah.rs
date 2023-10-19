#![feature(slice_flatten)]

use az_discrete_opt::{arr_map::par_update_costs, log::BasicLog, int_min_tree::INTMinTree};
use dfdx::{tensor::{AutoDevice, TensorFrom, Tensor2D, ZerosTensor, Tensor, AsArray, Trace, WithEmptyTape, SplitTape, PutTape}, prelude::{DeviceBuildExt, Linear, ReLU, Module, ModuleMut, ZeroGrads, cross_entropy_with_logits_loss, mse_loss, Optimizer}, optim::Adam, tensor_ops::{AdamConfig, WeightDecay, Backward}, shapes::{Rank2, Axis}};
use graph_state::achiche_hansen::AHState;

const N: usize = 32;
const E: usize = N * (N - 1) / 2;
type State = AHState<N, E>;

const ACTION: usize = 2 * E;
const STATE: usize = E + ACTION + 1;
type StateVec = [f32; STATE];

const BATCH: usize = 1;

const HIDDEN_1: usize = 1280;
const HIDDEN_2: usize = 1024;

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

type Tree = INTMinTree;

fn main() {
    const EPOCH: usize = 30;
    const EPISODES: usize = 4_000;

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
    
    // we initialize tensors to 0 and fill them as needed, minimizing allocations
    let mut state_tensor: Tensor<Rank2<BATCH, STATE>, _, _> = dev.zeros();
    let mut prediction_tensor: Tensor<Rank2<BATCH, HIDDEN_2>, _, _> = dev.zeros();
    let mut probs_tensor: Tensor<Rank2<BATCH, ACTION>, _, _> = dev.zeros();
    let mut observed_probabilities_tensor: Tensor<Rank2<BATCH, ACTION>, _, _> = dev.zeros();
    let mut observed_values_tensor: Tensor<Rank2<BATCH, 1>, _, _> = dev.zeros();
    // let root_tensor = dev.tensor(root_vecs.clone());
    // let mut prediction_tensor = core_model.forward(root_tensor);
    // let logits_tensor = logits_model.forward(prediction_tensor.clone());
    // let mut probs_tensor = logits_tensor.softmax::<Axis<1>>();
    // let mut value_tensor = value_model.forward(prediction_tensor.clone());
    // let predictions: [ActionVec; BATCH] = probs_tensor.array();
    
    // roots change across epochs
    let (mut root_states, mut root_vecs): ([State; BATCH], [StateVec; BATCH])
        = AHState::par_generate_batch(5, 0.5);
    let mut all_losses: Vec<(f32, f32)> = vec![];

    
    (1..=EPOCH).for_each(|epoch| {
        println!("==== EPOCH {epoch} ====");
        // todo! refactor without outparam
        let mut root_costs: [f32; BATCH] = [0.0f32; BATCH];
        par_update_costs(&mut root_costs, &root_states);
        state_tensor.copy_from(&root_vecs.flatten());
        prediction_tensor = core_model.forward(state_tensor.clone());
        probs_tensor = logits_model.forward(prediction_tensor.clone()).softmax::<Axis<1>>();
        let predictions: [ActionVec; BATCH] = probs_tensor.array();
        let mut trees: [Tree; BATCH] = par_plant_forest(&root_states, &predictions);
        let mut logs: [BasicLog; BATCH] = BasicLog::par_new_logs();
        
        let mut grads = core_model.alloc_grads();
        (1..=EPISODES).for_each(|episode| {
            println!("==== EPISODE {episode} ====");
            // states change during episodes
            let mut states: [State; BATCH] = root_states.clone();
            let mut vecs: [StateVec; BATCH] = root_vecs.clone();
            /* for a single tree, the transitions are built by selection actions from states
            * 1. from state s, we select action a only if
            */
            let transitions: [Trans; BATCH] = par_simulate_forest_once(&trees, &mut states, &mut vecs);
            par_update_logs(&mut logs, &transitions);
            state_tensor.copy_from(&vecs.flatten());
            let prediction_tensor = core_model.forward(state_tensor.clone());
            let probs = logits_model.forward(prediction_tensor.clone()).softmax::<Axis<1>>().array();
            let values = value_model.forward(prediction_tensor.clone()).array();
            /* the end state is either:
            * 1. terminal
            *  - this only occurs if the last action was not marked as Exhausted
            *  - we avoid encountering the same ex
            
            */
            par_insert_new_states(&mut trees, &transitions, &probs);
            par_update_state_data(&mut trees, &transitions, &values);
        });
        let mut probs: [ActionVec; BATCH] = [[0.0f32; ACTION]; BATCH];
        let mut values: [[f32; 1]; BATCH] = [[0.0f32; 1]; BATCH];
        par_set_observations(&trees, &mut probs, &mut values);
        observed_probabilities_tensor.copy_from(&probs.flatten());
        observed_values_tensor.copy_from(&values.flatten());
        
        // pass the root state vector into the core model
        let root_tensor = dev.tensor(root_vecs.clone());
        let predictions = core_model.forward(root_tensor.trace(grads));
        let (predictions, tape) = predictions.split_tape();
        // calculated cross entropy loss
        let logits = logits_model.forward(predictions.clone().put_tape(tape));
        let entropy_tensor = cross_entropy_with_logits_loss(logits.with_empty_tape(), observed_probabilities_tensor.clone());
        let (entropy_tensor, tape) = entropy_tensor.split_tape();
        let entropy = entropy_tensor.array();
        // do the same for L2 loss on the value function
        let values = value_model.forward(predictions.put_tape(tape));
        let mse_tensor = mse_loss(values.with_empty_tape(), observed_values_tensor.clone());
        let mse = mse_tensor.array();
        grads = mse_tensor.backward();
        all_losses.push((entropy, mse));
        println!("{all_losses:?}");
        opt.update(&mut core_model, &grads).expect("optimizer failed");
        // is this correct management of the tape?
        core_model.zero_grads(&mut grads);
        par_use_logged_roots(&mut root_states, &mut root_vecs, &mut logs, 5 * epoch);
        par_update_costs(&mut root_costs, &root_states);
    });
}

fn par_use_logged_roots<const BATCH: usize>(
    roots: &mut [State; BATCH],
    root_vecs: &mut [StateVec; BATCH],
    logs: &mut [BasicLog; BATCH],
    epoch: usize,
) {
    todo!()
}

fn par_set_observations<const BATCH: usize>(
    trees: &[Tree; BATCH],
    probs: &mut [ActionVec; BATCH],
    values: &mut [[f32; 1]; BATCH],
) {
    todo!()
}

fn par_insert_new_states<const BATCH: usize>(
    trees: &mut [Tree; BATCH],
    trans: &[Trans; BATCH],
    probs: &[ActionVec; BATCH],
    
) {
    todo!()
}

fn par_update_state_data<const BATCH: usize>(
    trees: &mut [Tree; BATCH],
    trans: &[Trans; BATCH],
    v: &[[f32; 1]],
) {
    todo!()
}

type ActionVec = [f32; ACTION];
struct Trans;

fn par_update_logs<const BATCH: usize>(
    logs: &mut [BasicLog; BATCH],
    transitions: &[Trans; BATCH],
) {
    todo!()
}

fn par_simulate_forest_once<const BATCH: usize>(
    trees: &[Tree; BATCH],
    states: &mut [State; BATCH],
    vecs: &mut [StateVec; BATCH],
) -> [Trans; BATCH] {
    todo!()
}

fn par_plant_forest<const BATCH: usize>(
    roots: &[State; BATCH],
    predictions: &[ActionVec; BATCH],
) -> [Tree; BATCH] {
    todo!()
}
