#![feature(slice_flatten)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]

use rayon::{iter::{IntoParallelIterator, ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator}, slice::ParallelSliceMut};
use tensorboard_writer::TensorboardWriter;

use std::{
    io::{BufWriter, Write},
    time::SystemTime,
};

use az_discrete_opt::{
    az_model::{add_dirichlet_noise, dfdx::TwoModels},
    int_min_tree::{state_data::UpperEstimateData, INTMinTree},
    learning_loop::{prediction::PredictionData, state::StateData, tree::TreeData, LearningLoop},
    log::ArgminData,
    path::{set::ActionSet, ActionPath},
    space::StateActionSpace,
    state::{cost::Cost, prohibit::WithProhibitions},
    tensorboard::{tf_path, Summarize},
};
use dfdx::prelude::*;
use graph_state::{
    rooted_tree::{prohibited_space::ProhibitedConstrainedRootedOrderedTree, RootedOrderedTree},
    simple_graph::connected_bitset_graph::Conjecture2Dot1Cost,
};

use chrono::prelude::*;

use eyre::WrapErr;

const N: usize = 20;
type Space = ProhibitedConstrainedRootedOrderedTree<N>;

type RawState = RootedOrderedTree<N>;
type S = WithProhibitions<RawState>;
type P = ActionSet;

type Tree = INTMinTree<P>;
type C = Conjecture2Dot1Cost;

const ACTION: usize = (N - 1) * (N - 2) / 2 - 1;
const GAIN: usize = 1;
const RAW_STATE: usize = (N - 1) * (N - 2) / 2 - 1;
const STATE: usize = RAW_STATE + ACTION;

const BATCH: usize = 256;

const HIDDEN_1: usize = 256;
const HIDDEN_2: usize = 256;

type Logits = (
    (Linear<STATE, HIDDEN_1>, ReLU),
    (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
    Linear<HIDDEN_2, ACTION>,
);

type Valuation = (
    (Linear<STATE, HIDDEN_1>, ReLU),
    (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
    Linear<HIDDEN_2, 1>,
);

fn main() -> eyre::Result<()> {
    let out_dir = tf_path().join("07-c21-tree").join(Utc::now().to_rfc3339());
    dbg!(&out_dir);
    let out_file = out_dir.join("tfevents-losses");
    // create the directory if it doesn't exist
    std::fs::create_dir_all(&out_dir).wrap_err("failed to create output directory")?;
    let writer = BufWriter::new(std::fs::File::create(out_file)?);
    let mut writer = TensorboardWriter::new(writer);
    writer.write_file_version()?;

    let dev = AutoDevice::default();
    let pi_config = AdamConfig {
        lr: 2e-2,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: Some(WeightDecay::L2(1e-6)), // Some(WeightDecay::Decoupled(1e-6)),
    };
    let g_config = AdamConfig {
        lr: 1e-2,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: Some(WeightDecay::L2(1e-6)), // Some(WeightDecay::Decoupled(1e-6)),
    };
    let models: TwoModels<Logits, Valuation, BATCH, STATE, ACTION, GAIN> =
        TwoModels::new(dev, pi_config, g_config);
    let upper_estimate = |estimate: UpperEstimateData| {
        let UpperEstimateData {
            n_s,
            n_sa,
            g_sa_sum,
            p_sa,
            depth: _,
        } = estimate;
        debug_assert_ne!(n_sa, 0);
        let n_s = n_s as f32;
        let n_sa = n_sa as f32;
        let c_puct = 1e1;
        let g_sa = g_sa_sum / n_sa;
        let u_sa = g_sa + c_puct * p_sa * (n_s.sqrt() / n_sa);
        u_sa
    };
    let default_prohibitions = |s: &RawState| {
        s.edge_indices_ignoring_0_1_and_last_vertex()
            .collect::<Vec<_>>()
    };
    let space = ProhibitedConstrainedRootedOrderedTree::<N>;
    let random_state = |_: usize| {
        let mut rng = rand::thread_rng();
        loop {
            let state = RawState::generate_constrained(&mut rng);
            let prohibited_actions = default_prohibitions(&state);
            let state = WithProhibitions::new(state.clone(), prohibited_actions);
            debug_assert!(
                !state.prohibited_actions.contains(&170),
                "state = {state:?}",
            );
            if !space.is_terminal(&state) {
                break state;
            }
        }
    };
    let cost = |s: &S| {
        debug_assert!(!s.prohibited_actions.contains(&170), "s = {s:?}");
        let cost = s.state.conjecture_2_1_cost();
        cost
    };
    let mut roots = (0..BATCH).into_par_iter().map(random_state).collect::<Vec<_>>();
    let mut states = roots.clone();
    let mut costs = roots.par_iter().map(cost).collect::<Vec<_>>();
    let mut vectors = vec![0.0; BATCH * Space::DIM];
    let mut state_data = StateData::new(&mut roots, &mut states, &mut costs, &mut vectors, Space::DIM);
    state_data.par_write_state_vecs(&space);
    let mut pi = vec![0.0; BATCH * ACTION];
    let mut g = vec![0.0; BATCH * GAIN];
    let mut predictions = PredictionData::new(&mut pi, &mut g);
    let add_noise = |_: usize, pi: &mut [f32]| {
        let mut rng = rand::thread_rng();
        const ALPHA: [f32; ACTION] = [0.03; ACTION];
        add_dirichlet_noise(&mut rng, pi, &ALPHA, 0.25);
    };
    let tree_data = TreeData::par_new(
        &space,
        add_noise,
        &mut predictions,
        &state_data,
        BATCH,
        ACTION,
    );
    let mut learning_loop: LearningLoop<Space, _, _, _> =
        LearningLoop::new(state_data, models, predictions, tree_data, ACTION, GAIN);
    todo!();
    // let mut global_argmin: ArgminData<C> = learning_loop
        // .par_argmin()
        // .map(|(s, c)| ArgminData::new(s, c.clone(), 0, 0))
        // .unwrap();
    let epochs: usize = 250;
    let episodes: usize = 800;

    let mut logits_mask = vec![0.0; ACTION * BATCH];

    for epoch in 0..epochs {
        println!("==== EPOCH: {epoch} ====");
        for episode in 1..=episodes {
            if episode % 100 == 0 {
                println!("==== EPISODE: {episode} ====");
            }
            learning_loop.par_roll_out_episode(&space, cost, upper_estimate);
            let episode_argmin = learning_loop.par_argmin().unwrap();
            // update the global argmin
            todo!();
            // if episode_argmin.1.evaluate() < global_argmin.cost().evaluate() {
            //     global_argmin =
            //         ArgminData::new(episode_argmin.0, episode_argmin.1.clone(), episode, epoch);
            //     println!("new min = {}", global_argmin.cost().evaluate());
            //     println!("argmin  = {global_argmin:?}");
            //     writer.write_summary(
            //         SystemTime::now(),
            //         (episodes * epoch + episode) as i64,
            //         global_argmin.cost().summary(),
            //     )?;
            //     writer.get_mut().flush()?;
            // }
        }
        let loss = if true {
            logits_mask.fill(f32::MIN);
            logits_mask.par_chunks_exact_mut(ACTION)
            .zip_eq(learning_loop.states.get_roots())
            .for_each(|(mask, root)| {
                space.action_indices(root)
                    .for_each(|a| mask[a] = 0.0);
            });
            learning_loop.par_update_model(&space, Some(&logits_mask))
        } else {
            learning_loop.par_update_model(&space, None)
        };
        // Write summaries to file.
        writer.write_summary(SystemTime::now(), epoch as i64, loss.summary())?;
        writer.get_mut().flush()?;
        todo!();
        // writer.write_summary(
        //     SystemTime::now(),
        //     (episodes * epoch + episodes) as i64,
        //     global_argmin.cost().summary(),
        // )?;
        writer.get_mut().flush()?;
        let modify_root = |i, t: &Tree, s: &mut S| {
            let nodes = t.unstable_sorted_nodes();
            let node = nodes[0];
            let (action, _) = node;
            space.follow(
                s,
                action
                    .actions_taken()
                    .map(|a| space.action(a)),
            );
            if space.is_terminal(s) {
                let prohibited_actions = default_prohibitions(&s.state);
                s.prohibited_actions.clear();
                s.prohibited_actions.extend(prohibited_actions);
                if space.is_terminal(s) {
                    *s = random_state(i);
                }
            }
        };
        let reset_root = |i, _: &Tree, r: &mut S| {
            *r = random_state(i);
        };
        match epoch % 4 {
            3 => learning_loop.par_reset_with_next_root(&space, reset_root, cost, add_noise),
            _ => learning_loop.par_reset_with_next_root(&space, modify_root, cost, add_noise),
        };
    }
    dbg!(&out_dir);
    Ok(())
}
