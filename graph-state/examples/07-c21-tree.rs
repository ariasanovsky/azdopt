#![feature(slice_flatten)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]

use rayon::prelude::*;

use std::{
    io::{Write, BufWriter},
    mem::MaybeUninit,
    path::{Path, PathBuf}, time::SystemTime,
};

use az_discrete_opt::{
    int_min_tree::{state_data::UpperEstimateData, INTMinTree, transition::INTTransition},
    log::ArgminData,
    path::{set::ActionSet, ActionPath},
    space::StateActionSpace,
    state::{cost::Cost, prohibit::WithProhibitions}, az_model::{add_dirichlet_noise, AzModel, dfdx::TwoModels}, learning_loop::{prediction::PredictionData, state::StateData, tree::TreeData, par_roll_out_episode},
};
use dfdx::prelude::*;
use graph_state::{simple_graph::connected_bitset_graph::Conjecture2Dot1Cost, rooted_tree::{prohibited_space::ProhibitedConstrainedRootedOrderedTree, RootedOrderedTree}};
use rand::rngs::ThreadRng;

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

use tensorboard_writer::{SummaryBuilder, TensorboardWriter};

fn main() -> eyre::Result<()> {
    // implementing tensorboard
    let out_dir = std::env::var("OUT_DIR")
        .map_or_else(
            |_| {
                std::env::var("CARGO_MANIFEST_DIR")
                    .map(|manifest_dir| Path::new(&manifest_dir).join("target"))
            },
            |out_dir| Ok(PathBuf::from(out_dir)),
        )
        .unwrap_or_else(|_| "/home/azdopt/graph-state/target".into())
        .join("07-c21-tree")
        .join(Utc::now().to_rfc3339());
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
    let mut models: TwoModels<Logits, Valuation, BATCH, STATE, ACTION, GAIN> = TwoModels::new(&dev, pi_config, g_config);
    let mut predictions = PredictionData::<BATCH, ACTION, GAIN>::new();
    
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
    
    // generate states
    let default_prohibitions = |s: &RawState| {
        s.edge_indices_ignoring_0_1_and_last_vertex().collect::<Vec<_>>()
    };

    let random_state = |rng: &mut ThreadRng| loop {
        let state = RawState::generate_constrained(rng);
        let prohibited_actions = default_prohibitions(&state);
        let state = WithProhibitions::new(state.clone(), prohibited_actions);
        debug_assert!(
            !state.prohibited_actions.contains(&170),
            "state = {state:?}",
        );
        if !Space::is_terminal(&state) {
            break state;
        }
    };
    // calculate costs
    let cost = |s: &S| {
        debug_assert!(
            !s.prohibited_actions.contains(&170),
            "s = {s:?}",
        );
        let cost = s.state.conjecture_2_1_cost();
        cost
    };
    let mut states = StateData::<BATCH, STATE, S, C>::par_new::<Space, _>(
        |_| rand::thread_rng(),
        |_, rng| random_state(rng),
        |s| cost(s),
    );

    let p_t: [P; BATCH] = core::array::from_fn(|_| P::new());
    let mut global_argmin: ArgminData<C> = (states.get_states(), states.get_costs()).into_par_iter().min_by(|(_, a), (_, b)| {
        a.evaluate().partial_cmp(&b.evaluate()).unwrap()
    }).map(|(s, c)| {
        ArgminData::new(s, c.clone(), 0, 0)
    }).unwrap();
    
    const ALPHA: [f32; ACTION] = [0.03; ACTION];
    let epochs: usize = 250;
    let episodes: usize = 800;
    let nodes: [Option<_>; BATCH] = core::array::from_fn(|_| None);
    
    let trees: [Tree; BATCH] = {
        let mut trees: [MaybeUninit<Tree>; BATCH] = MaybeUninit::uninit_array();
        (&mut trees, predictions.pi_mut(), states.get_costs(), states.get_states())
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, (t, pi_0_theta, c_0, s_0)| {
                    add_dirichlet_noise(rng, pi_0_theta, &ALPHA, 0.25);
                    t.write(Tree::new::<Space>(pi_0_theta, c_0.evaluate(), s_0));
                });
        unsafe { MaybeUninit::array_assume_init(trees) }
    };

    let mut trees: TreeData<256, ActionSet> = TreeData::new(trees, p_t, nodes);
    
    for epoch in 0..epochs {
        println!("==== EPOCH: {epoch} ====");
        // set state vectors
        states.reset_states();
        states.par_write_state_vecs::<Space>();
        states.par_write_state_costs(cost);
        models.write_predictions(states.get_vectors(), &mut predictions);
        (trees.trees_mut(), predictions.pi_mut(), states.get_costs(), states.get_states())
                .into_par_iter()
                .for_each_init(
                    || rand::thread_rng(),
                    |rng, (t, pi_0_theta, c_0, s_0)| {
                        add_dirichlet_noise(rng, pi_0_theta, &ALPHA, 0.25);
                        // t.
                        *t = Tree::new::<Space>(pi_0_theta, c_0.evaluate(), s_0);
                    });
        for episode in 1..=episodes {
            if episode % 100 == 0 {
                println!("==== EPISODE: {episode} ====");
            }
            par_roll_out_episode::<BATCH, STATE, ACTION, GAIN, Space, _, _>(&mut states, &mut models, &mut predictions, &mut trees, cost, upper_estimate);
            let episode_argmin = (states.get_states(), states.get_costs()).into_par_iter().min_by(|(_, a), (_, b)| {
                a.evaluate().partial_cmp(&b.evaluate()).unwrap()
            }).unwrap();
            // update the global argmin
            if episode_argmin.1.evaluate() < global_argmin.cost().evaluate() {
                global_argmin = ArgminData::new(episode_argmin.0, episode_argmin.1.clone(), episode, epoch);
                println!("new min = {}", global_argmin.cost().evaluate());
                println!("argmin  = {global_argmin:?}");
                let summ = SummaryBuilder::new()
                    .scalar("cost/cost", global_argmin.cost().evaluate())
                    .scalar("cost/lambda_1", global_argmin.cost().lambda_1 as _)
                    .scalar("cost/mu", global_argmin.cost().matching.len() as _)
                    .build();
                // Write summaries to file.
                writer.write_summary(SystemTime::now(), (episodes * epoch + episode) as i64, summ)?;
                writer.get_mut().flush()?;
            }
        }
        let (pi_t, g_t) = predictions.get_mut();
        (trees.trees(), pi_t, g_t)
            .into_par_iter()
            .for_each(|(t, p, v)| t.write_observations(p, v));
        states.par_write_state_vecs::<Space>();
        let (entropy, mse) = models.update_model(states.get_vectors(), &predictions);
        
        let summ = SummaryBuilder::new()
            .scalar("loss/entropy", entropy)
            .scalar("loss/mse", mse)
            .build();
        // Write summaries to file.
        writer.write_summary(SystemTime::now(), epoch as i64, summ)?;
        writer.get_mut().flush()?;

        let summ = SummaryBuilder::new()
            .scalar("cost/cost", global_argmin.cost().evaluate())
            .scalar("cost/lambda_1", global_argmin.cost().lambda_1 as _)
            .scalar("cost/mu", global_argmin.cost().matching.len() as _)
            .build();
        // Write summaries to file.
        writer.write_summary(SystemTime::now(), (episodes * epoch + episodes) as i64, summ)?;
        writer.get_mut().flush()?;
        let select_node = |_i, nodes: Vec<(_, _)>| {
            nodes[0].clone()
        };
        if epoch % 4 != 3 {
            (states.get_roots_mut(), trees.trees_mut()).into_par_iter().enumerate().for_each(|(i, (s, t))| {
                let nodes = t.unstable_sorted_nodes();
                let selected_node = select_node(i, nodes);
                Space::follow(s, selected_node.0.actions_taken::<Space>().map(|a| Space::from_index(*a)));
                if Space::is_terminal(s) {
                    let prohibited_actions = default_prohibitions(&s.state);
                    s.prohibited_actions.clear();
                    s.prohibited_actions.extend(prohibited_actions);
                    if Space::is_terminal(s) {
                        let mut rng = rand::thread_rng();
                        *s = random_state(&mut rng);
                    }
                }
            })
        } else {
            states.get_roots_mut().par_iter_mut().for_each(|s| {
                let mut rng = rand::thread_rng();
                *s = random_state(&mut rng);
            })
        }
    }
    dbg!(&out_dir);
    Ok(())
}
