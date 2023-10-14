use az_discrete_opt::{arr_map::par_update_costs, log::BasicLog};
// use dfdx::{tensor::AutoDevice, prelude::{DeviceBuildExt, Linear, ReLU}, optim::Adam, tensor_ops::{AdamConfig, WeightDecay}};
use graph_state::achiche_hansen::AHState;

const HIDDEN_1: usize = 256;
const HIDDEN_2: usize = 128;

const N: usize = 32;
const E: usize = N * (N - 1) / 2;
type State = AHState<N, E>;

const ACTION: usize = 2 * E;
const STATE: usize = E + ACTION + 1;
type StateVec = [f32; STATE];

const BATCH: usize = 64;

// type Core = (
//     (Linear<STATE, HIDDEN_1>, ReLU),
//     (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
//     // Linear<HIDDEN_2, PREDICTION>,
// );

// type Logits = (
//     Linear<HIDDEN_2, ACTION>,
//     // Softmax,
// );

// type Valuation = (
//     Linear<HIDDEN_2, 1>,
//     // Linear<HIDDEN_2, 3>,
// );

fn main() {
    const EPOCH: usize = 30;
    const EPISODES: usize = 4_000;

    // let dev = AutoDevice::default();
    // let mut core_model = dev.build_module::<Core, f32>();
    // let mut logits_model = dev.build_module::<Logits, f32>();
    // let mut value_model = dev.build_module::<Valuation, f32>();
    // let mut opt = Adam::new(
    //     &core_model,
    //     AdamConfig {
    //         lr: 1e-2,
    //         betas: [0.5, 0.25],
    //         eps: 1e-6,
    //         weight_decay: Some(WeightDecay::Decoupled(1e-2)),
    //     },
    // );
    
    // roots change across epochs
    let (mut epoch_root_states, mut epoch_root_vecs): ([State; BATCH], [StateVec; BATCH]) = AHState::par_generate_batch();
    // states change during episodes
    let mut episode_states: [State; BATCH] = epoch_root_states.clone();
    let mut all_losses: Vec<(f32, f32)> = vec![];

    // let root_tensor = dev.tensor(root_vecs.clone());
    // let mut prediction_tensor = core_model.forward(root_tensor);
    // let logits_tensor = logits_model.forward(prediction_tensor.clone());
    // let mut probs_tensor = logits_tensor.softmax::<Axis<1>>();
    // let mut value_tensor = value_model.forward(prediction_tensor.clone());
    // let predictions: [ActionVec; BATCH] = probs_tensor.array();
    

    (1..=EPOCH).for_each(|epoch| {
        println!("==== EPOCH {epoch} ====");
        let mut epoch_root_costs: [f32; BATCH] = [0.0f32; BATCH];
        par_update_costs(&mut epoch_root_costs, &epoch_root_states);
        let mut episode_logs: [BasicLog; BATCH] = BasicLog::par_new_logs();

        // let root_tensor = dev.tensor(root_vecs.clone());
        // let mut prediction_tensor = core_model.forward(root_tensor);
        // let logits_tensor = logits_model.forward(prediction_tensor.clone());
        // let mut probs_tensor = logits_tensor.softmax::<Axis<1>>();
        // let mut value_tensor = value_model.forward(prediction_tensor.clone());
        // let predictions: [ActionVec; BATCH] = probs_tensor.array();
        // let mut trees: [Tree; BATCH] = par_plant_forest(&roots, &predictions);

        // let mut grads = core_model.alloc_grads();
        (1..=EPISODES).for_each(|episode| {
            println!("==== EPISODE {episode} ====");
        });
    });
}
