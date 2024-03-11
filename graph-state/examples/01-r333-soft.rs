use std::{
    io::{BufWriter, Write},
    time::SystemTime,
};

use az_discrete_opt::{
    log::ArgminData,
    nabla::{
        model::dfdx::SoftActionModel,
        optimizer::{ArgminImprovement, NablaOptimizer, ResetPolicy},
        tree::graph_operations::ActionBudget,
    },
    path::set::ActionSet,
    tensorboard::{tf_path, Summarize},
};
use chrono::Utc;
use dfdx::{
    nn::{builders::Linear, modules::ReLU}, tensor::AutoDevice, tensor_ops::{AdamConfig, WeightDecay}
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
use rand::{rngs::ThreadRng, Rng};
use rand_distr::WeightedIndex;
use tensorboard_writer::TensorboardWriter;

const N: usize = 3;
const E: usize = N * (N - 1) / 2;
// const C: usize = 3;
// const SIZES: [usize; C] = [3, 3, 3];
const C: usize = 2;
const SIZES: [usize; C] = [3, 3];

type S = RamseyCountsNoRecolor<N, E, C, B32>;
type Cost = TotalCounts<C>;
type P = ActionSet;

type Space = RamseySpaceNoEdgeRecolor<B32, N, E, C>;

const ACTION: usize = E * C;
const STATE: usize = E * (2 * C + 1);

const HIDDEN_1: usize = 1024;
const HIDDEN_2: usize = 1024;
const HIDDEN_3: usize = 1024;
const HIDDEN_4: usize = 1024;
const HIDDEN_5: usize = 1024;

const LABELS: usize = 50;
const AL: usize = ACTION * LABELS;

type ModelH = (
    (Linear<STATE, HIDDEN_1>, ReLU),
    (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
    (Linear<HIDDEN_2, HIDDEN_3>, ReLU),
    (Linear<HIDDEN_3, HIDDEN_4>, ReLU),
    (Linear<HIDDEN_4, HIDDEN_5>, ReLU),
    // Linear<HIDDEN_5, {ACTION * LABELS}>,
    dfdx::nn::modules::SplitInto<(Linear<HIDDEN_5, {ACTION * LABELS}>, Linear<HIDDEN_5, ACTION>)>,
);

const BATCH: usize = 1;

type W = TensorboardWriter<BufWriter<std::fs::File>>;

fn main() -> eyre::Result<()> {
    let out_dir = tf_path().join("01-r333-soft").join(Utc::now().to_rfc3339());
    dbg!(&out_dir);
    let out_file = out_dir.join("tfevents-losses");
    // create the directory if it doesn't exist
    std::fs::create_dir_all(&out_dir).wrap_err("failed to create output directory")?;
    let writer = BufWriter::new(std::fs::File::create(out_file)?);
    let mut writer: W = TensorboardWriter::new(writer);
    writer.write_file_version()?;

    let dev = AutoDevice::default();
    // let num_permitted_edges_range = 8..=(4 * E / 5);
    // let reset_edges_range = (3 * E / 5)..=(4 * E / 5);
    let num_permitted_edges_range = E..=E;
    let reset_edges_range = E..=E;
    let dist = WeightedIndex::new([1., 0.,])?;
    let init_state = |rng: &mut ThreadRng, num_permitted_edges: usize| -> S {
        let g = ColoredCompleteBitsetGraph::generate(&dist, rng);
        let s = RamseyCounts::new(g, &SIZES);
        RamseyCountsNoRecolor::generate(rng, s, num_permitted_edges)
    };

    let cfg = AdamConfig {
        lr: 5e-3,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: Some(WeightDecay::L2(1e-6)),
    };
    let mut labels: [f32; LABELS] = core::array::from_fn(|i| i as f32 - 0.5);
    labels[49] = 100.;
    let model: SoftActionModel<ModelH, BATCH, STATE, ACTION, LABELS, AL> = SoftActionModel::new(dev, cfg, labels);
    // let model = az_discrete_opt::nabla::model::TrivialModel;
    const SPACE: Space = RamseySpaceNoEdgeRecolor::new(SIZES, [1., 1.,]);
    // let sample = SamplePattern {
    //     max: 2 * 10,
    //     mid: 2 * 7,
    //     min: 2 * 8,
    // };
    let budget = ActionBudget {
        g_budget: 10,
        p_budget: 10,
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
        &budget,
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
    let epochs: usize = 1; //250;
    let episodes: usize = 100 * 4 / 4;

    let n_as_tol = |len| {
        [200 / 5, 200 / 5, 200 / 5, 200 / 5, 200 / 5, 200 / 5, 200 / 5, 200 / 5, 200 / 5, 200 / 5, 100 / 5, 100 / 5, 100 / 5].get(len).copied().unwrap_or(50 / 5)
    };
    let n_obs_tol = 200 / 5;

    for epoch in 1..=epochs {
        println!("==== EPOCH: {epoch} ====");
        for episode in 1..=episodes {
            let new_argmin = optimizer.par_roll_out_episodes(n_as_tol, &budget);
            match new_argmin {
                ArgminImprovement::Improved(argmin) => {
                    let step = (episodes * (epoch - 1) + episode) as i64;
                    let _ = process_argmin(&argmin, &mut writer, step)?;
                }
                ArgminImprovement::Unchanged => {}
            };
            if true || episode  == episodes {
                println!("==== EPISODE: {episode} ====");
                // let sizes = optimizer
                //     .trees()
                //     .first()
                //     .unwrap()
                //     .sizes::<P>();
                // println!("sizes: {sizes:?}");

                let graph = optimizer.trees()[0].graphviz();
                std::fs::write("tree.svg", graph).unwrap();
                // let graph = optimizer.trees()[BATCH - 1].graphviz();
                // std::fs::write("tree2.svg", graph).unwrap();
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
        let reset_policy = ResetPolicy {
            greed: 0.5,
            adjust_improved_root: |s: &mut S| {
                let mut rng = rand::thread_rng();
                let num_permitted_edges = rng.gen_range(num_permitted_edges_range.clone());
                s.randomize_permitted_edges(num_permitted_edges, &mut rng);
            },
            adjust_unimproved_root: |s: &mut S| {
                let mut rng = rand::thread_rng();
                let num_permitted_edges = s.num_permitted_edges();
                if reset_edges_range.contains(&num_permitted_edges) {
                    let num_permitted_edges = rng.gen_range(num_permitted_edges_range.clone());
                    *s = init_state(&mut rng, num_permitted_edges);
                } else {
                    let range = num_permitted_edges..=*num_permitted_edges_range.end();
                    let num_permitted_edges = rng.gen_range(range);
                    s.randomize_permitted_edges(num_permitted_edges, &mut rng);
                }
            },
        };
        optimizer.par_reset_trees(reset_policy, &budget);
    }
    Ok(())
}
