use dfdx::{tensor::AutoDevice, prelude::{DeviceBuildExt, Linear, ReLU}, optim::Adam, tensor_ops::{AdamConfig, WeightDecay}};
use graph_state::achiche_hansen::AHState;

const HIDDEN_1: usize = 256;
const HIDDEN_2: usize = 128;

const N: usize = 32;
const E: usize = N * (N - 1) / 2;
pub type State = AHState<N, E>;


const STATE: usize = 2 * E;
const ACTION: usize = 2 * E;

const BATCH: usize = 64;

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

fn main() {
    const EPOCH: usize = 30;
    const EPISODES: usize = 4_000;

    let dev = AutoDevice::default();
    let mut core_model = dev.build_module::<Core, f32>();
    let mut logits_model = dev.build_module::<Logits, f32>();
    let mut value_model = dev.build_module::<Valuation, f32>();
    // let mut opt = Adam::new(
    //     &core_model,
    //     AdamConfig {
    //         lr: 1e-2,
    //         betas: [0.5, 0.25],
    //         eps: 1e-6,
    //         weight_decay: Some(WeightDecay::Decoupled(1e-2)),
    //     },
    // );

    let mut roots: [State; BATCH] = AHState::par_generate_batch();
    
}
