#![feature(slice_flatten)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]

use core::mem::MaybeUninit;
use std::array::from_fn;

use az_discrete_opt::{
    arr_map::par_set_costs,
    int_min_tree::{INTMinTree, INTTransitions},
    log::SimpleRootLog,
    state::{Reset, StateNode, StateVec},
};
use dfdx::{
    optim::Adam,
    prelude::{
        cross_entropy_with_logits_loss, mse_loss, DeviceBuildExt, Linear, Module, Optimizer, ReLU,
        ZeroGrads,
    },
    shapes::{Axis, Rank2},
    tensor::{AsArray, AutoDevice, Tensor, TensorFrom, Trace, ZerosTensor},
    tensor_ops::{AdamConfig, Backward, WeightDecay},
};
use graph_state::simple_graph::connected_bitset_graph::ConnectedBitsetGraph;
use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

const N: usize = 31;
const E: usize = N * (N - 1) / 2;
type State = ConnectedBitsetGraph<N>;
type Node = StateNode<State>;

const ACTION: usize = 2 * E;
const STATE: usize = E + ACTION + 1;
type StateVector = [f32; STATE];

const BATCH: usize = 1;

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

type Tree = INTMinTree;

fn main() {
    const EPOCH: usize = 100;
    const EPISODES: usize = 1_000;

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
    let mut v_t_tensor: Tensor<Rank2<BATCH, STATE>, _, _> = dev.zeros();
    let mut prediction_tensor: Tensor<Rank2<BATCH, HIDDEN_2>, _, _> = dev.zeros();
    let mut probs_tensor: Tensor<Rank2<BATCH, ACTION>, _, _> = dev.zeros();
    let mut observed_probabilities_tensor: Tensor<Rank2<BATCH, ACTION>, _, _> = dev.zeros();
    let mut observed_values_tensor: Tensor<Rank2<BATCH, 1>, _, _> = dev.zeros();
    // roots change across epochs
    let mut s_0 = from_fn(|_| {
        let mut rng = rand::thread_rng();
        StateNode::new(ConnectedBitsetGraph::generate(0.4, &mut rng), 30)
    });
    let mut all_losses: Vec<(f32, f32)> = vec![];

    // let cost = |s: &Node| s.state().num_edges() as f32;
    let cost = |s: &Node| s.state().ah_cost();
    (1..=EPOCH).for_each(|epoch| {
        // println!("==== EPOCH {epoch} ====");
        // todo! refactor without outparam
        let mut c_t: [f32; BATCH] = [0.0f32; BATCH];
        par_set_costs(&mut c_t, &s_0, &cost);
        let mut v_0 = from_fn(|_| [0.0f32; STATE]);
        par_set_vecs(&mut v_0, &s_0);

        v_t_tensor.copy_from(&v_0.flatten());
        prediction_tensor = core_model.forward(v_t_tensor.clone());
        probs_tensor = logits_model
            .forward(prediction_tensor.clone())
            .softmax::<Axis<1>>();
        let predictions: [ActionVec; BATCH] = probs_tensor.array();
        let mut trees: [Tree; BATCH] = par_plant_forest(&predictions, &c_t, &s_0);
        let mut logs: [SimpleRootLog<Node>; BATCH] = SimpleRootLog::par_new_logs(&s_0, &c_t);

        println!(
            "{}",
            logs[0]
                .next_root()
                .state()
                .to_graph6()
                .iter()
                .map(|&b| b as char)
                .collect::<String>()
        );

        let mut grads = core_model.alloc_grads();
        (1..=EPISODES).for_each(|episode| {
            if episode % 100 == 0 {
                // println!("==== EPISODE {episode} ====");
            }
            // states change during episodes
            let mut s_t: [Node; BATCH] = s_0.clone();
            let mut v_t: [StateVector; BATCH] = [[0.0f32; STATE]; BATCH];
            /* for a single tree, the transitions are built by selection actions from states
             * 1. from state s, we select action a only if
             */
            let transitions: [Trans; BATCH] = par_simulate_forest_once(&mut trees, &mut s_t);
            // todo! why is c_t unchanged? is s_t unchanged?
            // dbg!(&c_t);
            par_set_costs(&mut c_t, &s_t, &cost);
            // dbg!(&c_t);
            par_set_vecs(&mut v_t, &s_t);
            // panic!("{c_t:?}");
            par_update_logs(&mut logs, &s_t, &c_t);
            v_t_tensor.copy_from(v_t.flatten());
            let prediction_tensor = core_model.forward(v_t_tensor.clone());
            let probs = logits_model
                .forward(prediction_tensor.clone())
                .softmax::<Axis<1>>()
                .array();
            let values = value_model.forward(prediction_tensor.clone()).array();
            /* the end state is either:
            * 1. terminal
            *  - this only occurs if the last action was not marked as Exhausted
            *  - we avoid encountering the same ex

            */
            todo!(
                "
            par_insert_new_states(&mut trees, &transitions, &s_t, &c_t, &probs);
            par_update_state_data(&mut trees, &transitions, &c_t, &values);
            "
            );
        });
        let mut probs: [ActionVec; BATCH] = [[0.0f32; ACTION]; BATCH];
        let mut values: [[f32; 1]; BATCH] = [[0.0f32; 1]; BATCH];
        par_set_observations(&trees, &mut probs, &mut values);
        // dbg!(&values);
        observed_probabilities_tensor.copy_from(probs.flatten());
        observed_values_tensor.copy_from(values.flatten());

        // todo! retry this `optimizer failed: UnusedParams(UnusedTensors { ids: [UniqueId(0), UniqueId(1), UniqueId(2), UniqueId(3)] })`
        // // pass the root state vector into the core model
        // let root_tensor = dev.tensor(v_0.clone());
        // let predictions = core_model.forward(root_tensor.trace(grads));
        // let (predictions, tape) = predictions.split_tape();
        // // calculated cross entropy loss
        // let logits = logits_model.forward(predictions.clone().put_tape(tape));
        // let entropy_tensor = cross_entropy_with_logits_loss(logits.with_empty_tape(), observed_probabilities_tensor.clone());
        // let (entropy_tensor, tape) = entropy_tensor.split_tape();
        // let entropy = entropy_tensor.array();
        // // do the same for L2 loss on the value function
        // let values = value_model.forward(predictions.put_tape(tape));
        // let mse_tensor = mse_loss(values.with_empty_tape(), observed_values_tensor.clone());
        // let mse = mse_tensor.array();
        // grads = mse_tensor.backward();
        let entropy: f32;
        let root_tensor = dev.tensor(v_0.clone());
        let traced_predictions = core_model.forward(root_tensor.trace(grads));
        let predicted_logits = logits_model.forward(traced_predictions);
        let cross_entropy =
            cross_entropy_with_logits_loss(predicted_logits, observed_probabilities_tensor.clone());
        entropy = cross_entropy.array();
        grads = cross_entropy.backward();

        let mse: f32;
        let root_tensor = dev.tensor(v_0.clone());
        let traced_predictions = core_model.forward(root_tensor.trace(grads));
        let predicted_values = value_model.forward(traced_predictions);
        let square_error = mse_loss(predicted_values, observed_values_tensor.clone());
        mse = square_error.array();
        grads = square_error.backward();

        all_losses.push((entropy, mse));
        // println!("{all_losses:?}");
        opt.update(&mut core_model, &grads)
            .expect("optimizer failed");
        // is this correct management of the tape?
        core_model.zero_grads(&mut grads);

        par_reset_logs(&mut logs, &c_t, 30);
        par_update_roots(&mut s_0, &logs);
        par_set_vecs(&mut v_0, &s_0);
        par_set_costs(&mut c_t, &s_0, &cost);
        // todo!("update probs before replanting");
        // par_replant_forest(&mut trees, &probs, &c_t, &s_0);
    });
}

type Trans<'a> = INTTransitions<'a>;

fn par_update_roots(roots: &mut [Node], logs: &[SimpleRootLog<Node>]) {
    let roots = roots.par_iter_mut();
    let logs = logs.par_iter();
    roots.zip_eq(logs).for_each(|(r, l)| {
        r.clone_from(l.next_root());
        // r.reset(time);
    });
}

fn par_reset_logs(logs: &mut [SimpleRootLog<Node>], costs: &[f32], time: usize) {
    // todo!();
    let logs = logs.par_iter_mut();
    let costs = costs.par_iter();
    logs.zip_eq(costs)
        .for_each(|(l, c)| l.next_root_mut().reset(time));
}

fn par_set_vecs(vecs_old: &mut [StateVector], s_t: &[Node]) {
    // todo!();
    let vecs = vecs_old.par_iter_mut();
    let states = s_t.par_iter();
    vecs.zip_eq(states).for_each(|(v, s)| {
        // todo!();
        s.write_vec(v);
    });
}

// fn par_update_last_cost<const BATCH: usize>(
//     costs: &mut [f32; BATCH],
//     states: &[Node; BATCH],
// ) {
//     let old_c_t = costs.par_iter_mut();
//     let new_s_t = states.par_iter();
//     old_c_t.zip_eq(new_s_t).for_each(|(c, s)| {
//         // todo!();
//         *c = s.state().ah_cost()
//     });
// }

fn par_set_observations<const BATCH: usize>(
    trees: &[Tree; BATCH],
    probs: &mut [ActionVec; BATCH],
    values: &mut [[f32; 1]; BATCH],
) {
    let trees = trees.par_iter();
    let probs = probs.par_iter_mut();
    let values = values.par_iter_mut();
    trees
        .zip_eq(probs)
        .zip_eq(values)
        .for_each(|((t, p), v)| t.observe(p, v));
}

fn par_insert_new_states<const BATCH: usize>(
    tree: &mut [Tree; BATCH],
    trans: &[Trans; BATCH],
    s_t: &[Node; BATCH],
    c_t: &[f32; BATCH],
    probs_t: &[ActionVec; BATCH],
) {
    (tree, trans, s_t, c_t, probs_t)
        .into_par_iter()
        .for_each(|(t, trans, s, c, p)| {
            let n = todo!();
            t.insert_node_at_next_level(n)
        })
}

fn par_update_state_data<const BATCH: usize>(
    trees: &mut [Tree; BATCH],
    trans: &[Trans; BATCH],
    last_calculated_costs: &[f32; BATCH],
    values: &[[f32; 1]; BATCH],
) {
    (trees, trans, last_calculated_costs, values)
        .into_par_iter()
        .for_each(|(t, trans, c, v)| {
            todo!() // t.update(trans, c, v)
        })
}

type ActionVec = [f32; ACTION];

fn par_update_logs(logs: &mut [SimpleRootLog<Node>], s_t: &[Node], c_t: &[f32]) {
    (logs, s_t, c_t)
        .into_par_iter()
        .for_each(|(l, s, c)| l.update(s, c))
}

fn par_simulate_forest_once<'a, const BATCH: usize>(
    trees: &'a mut [Tree; BATCH],
    s_0: &mut [Node; BATCH],
) -> [Trans<'a>; BATCH] {
    let mut trans: [MaybeUninit<Trans>; BATCH] = MaybeUninit::uninit_array();
    (&mut trans, trees, s_0)
        .into_par_iter()
        .for_each(|(t, tree, s)| {
            t.write(tree.simulate_once(s));
        });
    unsafe { MaybeUninit::array_assume_init(trans) }
}

fn par_plant_forest<const BATCH: usize>(
    root_predictions: &[ActionVec; BATCH],
    costs: &[f32; BATCH],
    roots: &[Node; BATCH],
) -> [Tree; BATCH] {
    let mut trees: [MaybeUninit<Tree>; BATCH] = MaybeUninit::uninit_array();
    (&mut trees, root_predictions, costs, roots)
        .into_par_iter()
        .for_each(|(t, p, c, r)| {
            t.write(INTMinTree::new(p, *c, r));
        });
    unsafe { MaybeUninit::array_assume_init(trees) }
}
