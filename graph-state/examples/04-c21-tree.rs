#![feature(isqrt)]

use std::{
    io::{BufWriter, Write},
    time::SystemTime,
};

use az_discrete_opt::{
    log::ArgminData,
    nabla::{
        model::dfdx::ActionModel,
        optimizer::{ArgminImprovement, NablaOptimizer},
        space::DfaWithCost,
        tree::state_weight::StateWeight,
    },
    path::{set::ActionSet, ActionPath},
    tensorboard::{tf_path, Summarize},
};
use chrono::Utc;
use dfdx::{
    nn::{builders::Linear, modules::ReLU},
    tensor::AutoDevice,
    tensor_ops::{AdamConfig, WeightDecay},
};
use eyre::Context;
use graph_state::{
    rooted_tree::{modify_parent_once::ROTWithActionPermissions, space::ROTModifyParentsOnce},
    simple_graph::connected_bitset_graph::Conjecture2Dot1Cost,
};
use rand::{seq::SliceRandom, Rng};
use tensorboard_writer::TensorboardWriter;

const N: usize = 19;
const STATE: usize = (N - 1) * (N - 2) - 2;
const ACTION: usize = (N - 1) * (N - 2) / 2 - 1;

type S = ROTWithActionPermissions<N>;
type Cost = Conjecture2Dot1Cost;
type P = ActionSet;
type Space = ROTModifyParentsOnce<N, Cost>;

const HIDDEN_1: usize = 512;
const HIDDEN_2: usize = 1024;
const HIDDEN_3: usize = 512;

type ModelH = (
    (Linear<STATE, HIDDEN_1>, ReLU),
    (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
    (Linear<HIDDEN_2, HIDDEN_3>, ReLU),
    (Linear<HIDDEN_3, ACTION>, dfdx::nn::modules::Sigmoid),
    // (Linear<HIDDEN_3, ACTION>, ReLU),
);

const BATCH: usize = 512;

type W = TensorboardWriter<BufWriter<std::fs::File>>;

const C_LOWER_BOUND: usize = 2;
const C_UPPER_BOUND: usize = {
    const _SQRT: usize = (N - 1).isqrt();
    const SQRT: usize = if _SQRT * _SQRT == N - 1 {
        _SQRT
    } else {
        _SQRT + 1
    };
    const MU_MAX: usize = (N + 1) / 2;
    SQRT + MU_MAX
};

fn squish(x: f32) -> f32 {
    const SLOPE: f32 = 1.0 / ((C_UPPER_BOUND - C_LOWER_BOUND) as f32);
    let x = x - C_LOWER_BOUND as f32;
    SLOPE * x
}

fn main() -> eyre::Result<()> {
    let out_dir = tf_path().join("04-c21-tree").join(Utc::now().to_rfc3339());
    dbg!(&out_dir);
    let out_file = out_dir.join("tfevents-losses");
    std::fs::create_dir_all(&out_dir).wrap_err("failed to create output directory")?;
    let writer = BufWriter::new(std::fs::File::create(out_file)?);
    let mut writer: W = TensorboardWriter::new(writer);
    writer.write_file_version()?;

    let num_permitted_actions_range = 5..=(ACTION / 2);
    // let num_permitted_actions_range = 3..=3;
    let cfg = AdamConfig {
        lr: 1e-4,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: Some(WeightDecay::L2(1e-6)),
    };
    let model: ActionModel<ModelH, BATCH, STATE, ACTION> =
        ActionModel::new(AutoDevice::default(), cfg);
    // let model = az_discrete_opt::nabla::model::TrivialModel;
    const SPACE: Space = Space::new(
        |t| t.conjecture_2_1_cost(),
        |c| {
            let Conjecture2Dot1Cost { matching, lambda_1 } = c;
            let c = matching.len() as f32 + *lambda_1 as f32;
            squish(c)
        },
        |c_s, h_theta_sa| c_s - h_theta_sa,
        |_c_s, c_as_star| c_as_star,
    );
    let mut optimizer: NablaOptimizer<_, _, P> = NablaOptimizer::par_new(
        SPACE,
        || {
            let mut rng = rand::thread_rng();
            let num_permitted_actions = rng.gen_range(num_permitted_actions_range.clone());
            ROTWithActionPermissions::generate(&mut rng, num_permitted_actions)
        },
        model,
        BATCH,
    );

    let goal: f32 = squish(5.2);

    let process_argmin = |argmin: &ArgminData<S, Cost>, writer: &mut W, step: i64| {
        let ArgminData { state, cost, eval } = argmin;
        println!("{eval:12}\t{cost:?}");
        writer.write_summary(SystemTime::now(), step, cost.summary())?;
        writer.get_mut().flush()?;

        if *eval < goal {
            Err(eyre::eyre!(format!("state is optimal:\n{state}")))
        } else {
            Ok(())
        }
    };
    let argmin = optimizer.argmin_data();
    process_argmin(&argmin, &mut writer, 0)?;
    let epochs: usize = 250;
    let episodes: usize = 800;
    let n_obs_tol = 200;
    let n_as_tol = |len: usize| -> u32 {
        [200, 50, 50].get(len).copied().unwrap_or(25)
    };

    for epoch in 1..=epochs {
        println!("==== EPOCH: {epoch} ====");
        for episode in 1..=episodes {
            let new_argmin = optimizer.par_roll_out_episodes(n_as_tol);
            match new_argmin {
                ArgminImprovement::Improved(argmin) => {
                    let step = (episodes * (epoch - 1) + episode) as i64;
                    let _ = process_argmin(&argmin, &mut writer, step)?;
                }
                ArgminImprovement::Unchanged => {}
            };
            if episode % episodes == 0 {
                println!("==== EPISODE: {episode} ====");
                let sizes = optimizer.trees().first().unwrap().sizes();
                println!("sizes: {sizes:?}");
                let graph = optimizer.trees()[0].graphviz();
                std::fs::write("tree.png", graph).unwrap();
                let graph = optimizer.trees()[BATCH - 1].graphviz();
                std::fs::write("tree2.png", graph).unwrap();
                // panic!();
            }
        }

        let loss = optimizer.par_update_model(n_obs_tol);
        let summary = loss.summary();
        writer.write_summary(SystemTime::now(), (episodes * epoch) as i64, summary)?;
        writer.write_summary(
            SystemTime::now(),
            (episodes * epoch) as i64,
            optimizer.argmin_data().cost.summary(),
        )?;
        writer.get_mut().flush()?;
        let modify_root = |space: &Space, state: &mut S, mut n: Vec<(&P, &StateWeight)>| {
            let mut rng = rand::thread_rng();
            let (_, state_weight) = n[0];
            let c_root = state_weight.c();
            let c_root_star = state_weight.c_star();
            if c_root == c_root_star {
                let num_permitted_actions = state.permitted_actions.len();
                if !num_permitted_actions_range.contains(&num_permitted_actions) {
                    unreachable!();
                } else if num_permitted_actions == *num_permitted_actions_range.end() {
                    let num_prohibitions = rng.gen_range(num_permitted_actions_range.clone());
                    *state = ROTWithActionPermissions::generate(&mut rng, num_prohibitions);
                } else {
                    n.retain(|(_, w)| w.c() == c_root);
                    let (p, _) = n.choose(&mut rng).unwrap();
                    p.actions_taken().for_each(|a| {
                        let a = space.action(a);
                        space.act(state, &a);
                    });
                    let range = num_permitted_actions..=*num_permitted_actions_range.end();
                    let num_permitted_edges = rng.gen_range(range);
                    state.randomize_permitted_actions(&mut rng, num_permitted_edges);
                }
            } else {
                let c_threshold = (c_root + 3.0 * c_root_star) / 4.0;
                n.retain(|(_, w)| w.c() <= c_threshold);
                let (p, _) = n.choose(&mut rng).unwrap();
                p.actions_taken().for_each(|a| {
                    let a = space.action(a);
                    space.act(state, &a);
                });
                let num_permitted_actions = rng.gen_range(num_permitted_actions_range.clone());
                state.randomize_permitted_actions(&mut rng, num_permitted_actions);
            }
        };
        optimizer.par_reset_trees(modify_root);
    }

    Ok(())
}
