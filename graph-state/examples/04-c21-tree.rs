use std::{io::{BufWriter, Write}, time::SystemTime};

use az_discrete_opt::{path::{set::ActionSet, ActionPath}, tensorboard::{tf_path, Summarize}, nabla::{model::dfdx::ActionModel, optimizer::{NablaOptimizer, ArgminImprovement}, space::NablaStateActionSpace}, log::ArgminData};
use chrono::Utc;
use dfdx::{nn::{modules::ReLU, builders::Linear}, tensor::AutoDevice, tensor_ops::{AdamConfig, WeightDecay}};
use eyre::Context;
use graph_state::{rooted_tree::{modify_parent_once::ROTWithActionPermissions, space::ROTModifyParentsOnce}, simple_graph::connected_bitset_graph::Conjecture2Dot1Cost};
use rand::{Rng, seq::SliceRandom};
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

const BATCH: usize = 1;

type W = TensorboardWriter<BufWriter<std::fs::File>>;

fn main() -> eyre::Result<()> {
    let out_dir = tf_path().join("04-c21-tree").join(Utc::now().to_rfc3339());
    dbg!(&out_dir);
    let out_file = out_dir.join("tfevents-losses");
    std::fs::create_dir_all(&out_dir).wrap_err("failed to create output directory")?;
    let writer = BufWriter::new(std::fs::File::create(out_file)?);
    let mut writer: W = TensorboardWriter::new(writer);
    writer.write_file_version()?;

    let num_permitted_actions_range = 3..=3;
    let cfg = AdamConfig {
        lr: 5e-3,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: Some(WeightDecay::L2(1e-6)),
    };
    let model: ActionModel<ModelH, BATCH, STATE, ACTION> = ActionModel::new(AutoDevice::default(), cfg);
    // let model = az_discrete_opt::nabla::model::TrivialModel;
    const SPACE: Space = Space::new(
        |t| {
            t.conjecture_2_1_cost()
        },
        |c| {
            let Conjecture2Dot1Cost { matching, lambda_1 } = c;
            matching.len() as f32 + *lambda_1 as f32
        }
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

    let process_argmin = |argmin: &ArgminData<S, Cost>, writer: &mut W, step: i64| {
        let ArgminData { state, cost, eval } = argmin;
        // println!("{eval:12}\t{cost:?}");
        writer.write_summary(SystemTime::now(), step, cost.summary())?;
        writer.get_mut().flush()?;

        if *eval < 5.2 {
            Err(eyre::eyre!(format!("state is optimal:\n{state}")))
        } else {
            Ok(())
        }
    };
    let argmin = optimizer.argmin_data();
    process_argmin(&argmin, &mut writer, 0)?;
    let epochs: usize = 250;
    let episodes: usize = 3_200;

    let decay = 1.1;

    for epoch in 1..=epochs {
        println!("==== EPOCH: {epoch} ====");
        for episode in 1..=episodes {
            let new_argmin = optimizer.par_roll_out_episodes(decay);
            match new_argmin {
                ArgminImprovement::Improved(argmin) => {
                    let step = (episodes * (epoch - 1) + episode) as i64;
                    let _ = process_argmin(&argmin, &mut writer, step)?;
                }
                ArgminImprovement::Unchanged => {}
            };
            if episode % episodes == 0 {
                println!("==== EPISODE: {episode} ====");
                let sizes = optimizer
                    .get_trees()
                    .first()
                    .unwrap()
                    .sizes()
                    .collect::<Vec<_>>();
                println!("sizes: {sizes:?}");

                let graph = optimizer.get_trees()[0].graphviz();
                std::fs::write("tree.png", graph).unwrap();
                let graph = optimizer.get_trees()[BATCH - 1].graphviz();
                std::fs::write("tree2.png", graph).unwrap();
            }
        }

        let loss = optimizer.par_update_model();
        let summary = loss.summary();
        writer.write_summary(SystemTime::now(), (episodes * epoch) as i64, summary)?;
        writer.write_summary(
            SystemTime::now(),
            (episodes * epoch) as i64,
            optimizer.argmin_data().cost.summary(),
        )?;
        writer.get_mut().flush()?;
        let modify_root = |space: &Space, state: &mut S, mut n: Vec<(Option<&P>, f32, f32)>| {
            let mut rng = rand::thread_rng();
            let (_, c_root, c_root_star) = n[0];
            if c_root == c_root_star {
                let num_permitted_actions = state.permitted_actions.len();
                if !num_permitted_actions_range.contains(&num_permitted_actions) {
                    unreachable!();
                } else if num_permitted_actions == *num_permitted_actions_range.end() {
                    let num_prohibitions = rng.gen_range(num_permitted_actions_range.clone());
                    *state = ROTWithActionPermissions::generate(&mut rng, num_prohibitions);
                } else {
                    n.retain(|(_, c, _)| *c == c_root);
                    let (p, _, _) = n.choose(&mut rng).unwrap();
                    if let Some(p) = p {
                        p.actions_taken().for_each(|a| {
                            let a = space.action(a);
                            space.act(state, &a);
                        })
                    }
                    let range = num_permitted_actions..=*num_permitted_actions_range.end();
                    let num_permitted_edges = rng.gen_range(range);
                    state.randomize_permitted_actions(&mut rng, num_permitted_edges);
                }
            } else {
                n.retain(|(_, c, _)| *c == c_root_star);
                let (p, _, _) = n.choose(&mut rng).unwrap();
                if let Some(p) = p {
                    p.actions_taken().for_each(|a| {
                        let a = space.action(a);
                        space.act(state, &a);
                    })
                }
                let num_permitted_actions = rng.gen_range(num_permitted_actions_range.clone());
                state.randomize_permitted_actions(&mut rng, num_permitted_actions);
            }
        };
        optimizer.par_reset_trees(modify_root);
    }
    
    Ok(())
}
