use std::{
    io::{BufWriter, Write},
    time::SystemTime,
};

use az_discrete_opt::{
    log::ArgminData,
    nabla::{
        model::dfdx::ActionModel,
        optimizer::{ArgminImprovement, NablaOptimizer},
        space::NablaStateActionSpace, tree::{graph_operations::SamplePattern, state_weight::StateWeight},
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
    bitset::primitive::B32,
    ramsey_counts::{
        no_recolor::RamseyCountsNoRecolor, space::RamseySpaceNoEdgeRecolor, RamseyCounts,
        TotalCounts,
    },
    simple_graph::bitset_graph::ColoredCompleteBitsetGraph,
};
use rand::{rngs::ThreadRng, seq::SliceRandom, Rng};
use rand_distr::WeightedIndex;
use tensorboard_writer::TensorboardWriter;

const N: usize = 16;
const E: usize = N * (N - 1) / 2;
const C: usize = 3;
const SIZES: [usize; C] = [3, 3, 3];

type S = RamseyCountsNoRecolor<N, E, C, B32>;
type Cost = TotalCounts<C>;
type P = ActionSet;

type Space = RamseySpaceNoEdgeRecolor<B32, N, E, C>;

const ACTION: usize = E * C;
const STATE: usize = E * (2 * C + 1);

const HIDDEN_1: usize = 512;
const HIDDEN_2: usize = 1024;
const HIDDEN_3: usize = 1024;
const HIDDEN_4: usize = 512;

type ModelH = (
    (Linear<STATE, HIDDEN_1>, ReLU),
    (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
    (Linear<HIDDEN_2, HIDDEN_3>, ReLU),
    // (Linear<HIDDEN_3, HIDDEN_4>, ReLU),
    // (Linear<HIDDEN_4, ACTION>, dfdx::nn::modules::Sigmoid),
    (Linear<HIDDEN_3, ACTION>, ReLU),
);

const BATCH: usize = 1024;

type W = TensorboardWriter<BufWriter<std::fs::File>>;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() -> eyre::Result<()> {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let out_dir = tf_path().join("01-r333-grad").join(Utc::now().to_rfc3339());
    dbg!(&out_dir);
    let out_file = out_dir.join("tfevents-losses");
    // create the directory if it doesn't exist
    std::fs::create_dir_all(&out_dir).wrap_err("failed to create output directory")?;
    let writer = BufWriter::new(std::fs::File::create(out_file)?);
    let mut writer: W = TensorboardWriter::new(writer);
    writer.write_file_version()?;

    let dev = AutoDevice::default();
    let num_permitted_edges_range = 8..=(4 * E / 5);
    let dist = WeightedIndex::new([1., 1., 1.,])?;
    let init_state = |rng: &mut ThreadRng, num_permitted_edges: usize| -> S {
        let g = ColoredCompleteBitsetGraph::generate(&dist, rng);
        let s = RamseyCounts::new(g, &SIZES);
        RamseyCountsNoRecolor::generate(rng, s, num_permitted_edges)
    };

    let cfg = AdamConfig {
        lr: 1e-4,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: Some(WeightDecay::L2(1e-6)),
    };
    let model: ActionModel<ModelH, BATCH, STATE, ACTION> = ActionModel::new(dev, cfg);
    // let model = az_discrete_opt::nabla::model::TrivialModel;
    const SPACE: Space = RamseySpaceNoEdgeRecolor::new(SIZES, [1., 1., 1.,]);
    let sample = SamplePattern {
        max: 2 * 10,
        mid: 2 * 7,
        min: 2 * 8,
    };

    let mut optimizer: NablaOptimizer<_, _, P> = NablaOptimizer::par_new(
        SPACE,
        || {
            let mut rng = rand::thread_rng();
            let num_permitted_edges = rng.gen_range(num_permitted_edges_range.clone());
            init_state(&mut rng, num_permitted_edges)
        },
        model,
        BATCH,
        &sample,
    );
    let process_argmin = |argmin: &ArgminData<S, Cost>, writer: &mut W, step: i64| {
        let ArgminData { state, cost, eval } = argmin;
        println!("{eval}\t{cost:?}");
        writer.write_summary(SystemTime::now(), step, cost.summary())?;
        writer.get_mut().flush()?;

        if *eval == 0. {
            Err(eyre::eyre!(format!("state is optimal:\n{state}")))
        } else {
            Ok(())
        }
    };
    let argmin = optimizer.argmin_data();
    process_argmin(&argmin, &mut writer, 0)?;
    let epochs: usize = 250;
    let episodes: usize = 800;

    let n_as_tol = |len| {
        [200, 200, 200, 200, 100, 100, 100, 100].get(len).copied().unwrap_or(50)
    };
    let n_obs_tol = 200;

    for epoch in 1..=epochs {
        println!("==== EPOCH: {epoch} ====");
        for episode in 1..=episodes {
            let new_argmin = optimizer.par_roll_out_episodes(n_as_tol, &sample);
            match new_argmin {
                ArgminImprovement::Improved(argmin) => {
                    let step = (episodes * (epoch - 1) + episode) as i64;
                    let _ = process_argmin(&argmin, &mut writer, step)?;
                }
                ArgminImprovement::Unchanged => {}
            };
            if episode  == episodes {
                println!("==== EPISODE: {episode} ====");
                let sizes = optimizer
                    .get_trees()
                    .first()
                    .unwrap()
                    .sizes();
                println!("sizes: {sizes:?}");

                let graph = optimizer.get_trees()[0].graphviz();
                std::fs::write("tree.png", graph).unwrap();
                let graph = optimizer.get_trees()[BATCH - 1].graphviz();
                std::fs::write("tree2.png", graph).unwrap();
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
                let num_permitted_edges = state.num_permitted_edges();
                if !num_permitted_edges_range.contains(&num_permitted_edges) {
                    unreachable!();
                } else if num_permitted_edges == *num_permitted_edges_range.end() {
                    let num_prohibitions = rng.gen_range(num_permitted_edges_range.clone());
                    let counts = RamseyCounts::generate(&mut rng, &dist, &SIZES);
                    *state = RamseyCountsNoRecolor::generate(&mut rng, counts, num_prohibitions);
                } else {
                    n.retain(|(_, w)| w.c() == c_root);
                    let (p, _) = n.choose(&mut rng).unwrap();
                    p.actions_taken().for_each(|a| {
                        let a = space.action(a);
                        space.act(state, &a);
                    });
                    let range = num_permitted_edges..=*num_permitted_edges_range.end();
                    let num_permitted_edges = rng.gen_range(range);
                    state.randomize_permitted_edges(num_permitted_edges, &mut rng);
                }
            } else {
                let c_threshold = (c_root + 7.0 * c_root_star) / 8.0;
                n.retain(|(_, w)| w.c() <= c_threshold);
                let (p, _) = n.choose(&mut rng).unwrap();
                p.actions_taken().for_each(|a| {
                    let a = space.action(a);
                    space.act(state, &a);
                });
                let num_permitted_edges = rng.gen_range(num_permitted_edges_range.clone());
                state.randomize_permitted_edges(num_permitted_edges, &mut rng);
            }
        };
        optimizer.par_reset_trees(modify_root, &sample);
    }
    Ok(())
}
