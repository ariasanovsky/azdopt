use az_discrete_opt::arr_map::{par_plant_forest, par_insert_into_forest, par_forest_observations, par_update_costs, par_simulate_forest_once, par_update_forest, par_update_roots, par_state_batch_to_vecs};
use az_discrete_opt::ir_min_tree::{IRMinTree, Transitions};
use az_discrete_opt::log::{GraphLogs, par_update_logs};
use dfdx::optim::Adam;
use dfdx::prelude::{
    cross_entropy_with_logits_loss, mse_loss, DeviceBuildExt, Linear, Module, Optimizer, ReLU,
    ZeroGrads,
};
use dfdx::shapes::Axis;
use dfdx::tensor::{AsArray, AutoDevice, TensorFrom, Trace};
use dfdx::tensor_ops::{AdamConfig, Backward, WeightDecay};
use ramsey::ramsey_state::{ActionVec, GraphState, StateVec, ValueVec, STATE};
use ramsey::{C, E};

const ACTION: usize = C * E;
const BATCH: usize = 64;

type Tree = IRMinTree;

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
    let mut states: [GraphState; BATCH] = roots.clone();
    let mut root_costs: [f32; BATCH] =[0.0f32; BATCH];
    par_update_costs(&mut root_costs, &roots);
    let mut losses: Vec<(f32, f32)> = vec![];

    (1..=EPOCH).for_each(|epoch| {
        println!("==== EPOCH {epoch} ====");
        let mut grads = core_model.alloc_grads();
        let root_vecs: [StateVec; BATCH] = GraphState::par_generate_vecs(&roots);
        // dbg!(root_vecs.get(0).unwrap());

        let root_tensor = dev.tensor(root_vecs.clone());
        let mut prediction_tensor = core_model.forward(root_tensor);
        let logits_tensor = logits_model.forward(prediction_tensor.clone());
        let mut probs_tensor = logits_tensor.softmax::<Axis<1>>();
        let mut value_tensor = value_model.forward(prediction_tensor.clone());
        let predictions: [ActionVec; BATCH] = probs_tensor.array();
        let mut trees: [Tree; BATCH] = par_plant_forest(&roots, &predictions);

        let mut logs: [GraphLogs; 64] = GraphLogs::par_new_logs();

        // play episodes
        (1..=EPISODES).for_each(|episode| {
            let transitions: [Trans; BATCH] = par_simulate_forest_once(&trees, &roots, &mut states);
            par_update_logs(&mut logs, &transitions);
            if episode % 500 == 0 {
                println!("episode {episode}");
                transitions.chunks_exact(4).into_iter()
                    .zip(root_costs.chunks_exact(4).into_iter())
                    .for_each(|(trans, costs)| {
                        trans.iter().zip(costs).for_each(|(trans, cost)| {
                            let costs = trans.costs(*cost);
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
            let end_state_vecs: [StateVec; BATCH] = par_state_batch_to_vecs(&states);
            prediction_tensor = core_model.forward(dev.tensor(end_state_vecs));
            let logits_tensor = logits_model.forward(prediction_tensor.clone());
            probs_tensor = logits_tensor.clone().softmax::<Axis<1>>();
            value_tensor = value_model.forward(prediction_tensor.clone());
            let probs = probs_tensor.array();
            // let max_deviation_from_one = probs.iter().map(|probs| probs.into_iter().sum::<f32>()).map(|psum| (1.0f32 - psum).abs()).max_by(|a, b| a.partial_cmp(b).unwrap()).unwrap();
            // println!("max_deviation_from_one = {}", max_deviation_from_one);
            let values = value_tensor.array();
            par_update_forest(&mut trees, &transitions, &values);
            par_insert_into_forest(&mut trees, &transitions, &states, &probs);
        });
        // println!();
        // backprop loss
        let observations: ([ActionVec; BATCH], [ValueVec; BATCH]) = par_forest_observations(&trees);
        let entropy;
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
        par_update_costs(&mut root_costs, &roots);
    });
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
