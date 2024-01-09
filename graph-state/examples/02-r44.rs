use std::{io::{BufWriter, Write}, time::SystemTime};

use az_discrete_opt::{tensorboard::{tf_path, Summarize}, state::{prohibit::WithProhibitions, layers::Layers}, space::layered::Layered, log::ArgminData, nabla::{optimizer::{NablaOptimizer, ArgminImprovement}, model::dfdx::ActionModel, space::NablaStateActionSpace}, path::{set::ActionSet, ActionPath}};
use chrono::Utc;
use dfdx::{tensor::AutoDevice, tensor_ops::{AdamConfig, WeightDecay}, nn::{modules::ReLU, builders::Linear}};
use eyre::Context;
use graph_state::{bitset::primitive::B32, ramsey_counts::{RamseyCounts, space::RamseySpaceNoEdgeRecolor}, simple_graph::bitset_graph::ColoredCompleteBitsetGraph};
use rand::{seq::SliceRandom, rngs::ThreadRng, Rng};
use tensorboard_writer::TensorboardWriter;

const N: usize = 17;
const E: usize = N * (N - 1) / 2;
const C: usize = 2;
const SIZES: [usize; C] = [4, 4];

type RawState = RamseyCounts<N, E, C, B32>;
type S = WithProhibitions<RawState>;

type P = ActionSet;

type Space = RamseySpaceNoEdgeRecolor<B32, N, E, C>;

const ACTION: usize = E * C;
const STATE: usize = 3 * C * E;

const HIDDEN_1: usize = 512;
const HIDDEN_2: usize = 1024;
const HIDDEN_3: usize = 512;

type ModelH = (
    (Linear<STATE, HIDDEN_1>, ReLU),
    (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
    (Linear<HIDDEN_2, HIDDEN_3>, ReLU),
    // (Linear<HIDDEN_3, ACTION>, dfdx::nn::modules::Sigmoid),
    (Linear<HIDDEN_3, ACTION>, ReLU),
);

const BATCH: usize = 128;

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() -> eyre::Result<()> {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let out_dir = tf_path().join("01-r44-grad").join(Utc::now().to_rfc3339());
    dbg!(&out_dir);
    let out_file = out_dir.join("tfevents-losses");
    // create the directory if it doesn't exist
    std::fs::create_dir_all(&out_dir).wrap_err("failed to create output directory")?;
    let writer = BufWriter::new(std::fs::File::create(out_file)?);
    let mut writer = TensorboardWriter::new(writer);
    writer.write_file_version()?;

    let edges: [usize; E] = core::array::from_fn(|i| i);

    let dev = AutoDevice::default();
    let dist = rand::distributions::WeightedIndex::new([1., 1.])?;
    let new_state = |s: _, p: _| -> S {
        let s = WithProhibitions {
            state: s,
            prohibited_actions: p,
        };
        s
    };
    let init_state = |rng: &mut ThreadRng, num_prohibitions: usize| -> S {
        let g = ColoredCompleteBitsetGraph::generate(&dist, rng);
        let s = RamseyCounts::new(g, &SIZES);
        let p = (edges).choose_multiple(rng, num_prohibitions).flat_map(|i| {
            (0..C).map(|c| c * E + *i)
        }).collect();
        new_state(s, p)
    };

    let cfg = AdamConfig {
        lr: 3e-4,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: Some(WeightDecay::L2(1e-6)),
    };
    let model: ActionModel<ModelH, BATCH, STATE, ACTION> = ActionModel::new(dev, cfg);
    // let model = az_discrete_opt::nabla::model::TrivialModel;
    const SPACE: Space = RamseySpaceNoEdgeRecolor::new(SIZES, [1., 1.]);
    
    let mut optimizer: NablaOptimizer<_, _, P> = NablaOptimizer::par_new(
        SPACE, 
        || {
            let mut rng = rand::thread_rng();
            let num_prohibitions = rng.gen_range(0..E);
            init_state(&mut rng, num_prohibitions)
        },
        model,
        BATCH,
    );
    let ArgminData { state, cost, eval} = optimizer.argmin_data();
    writer.write_summary(SystemTime::now(), 0, cost.summary())?;
    writer.get_mut().flush()?;
    
    let epochs: usize = 250;
    let episodes: usize = 800;

    let decay = 0.4;

    for epoch in 1..=epochs {
        println!("==== EPOCH: {epoch} ====");
        for episode in 1..=episodes {
            let new_argmin = optimizer.par_roll_out_episodes(decay);
            match new_argmin {
                ArgminImprovement::Improved(ArgminData { state, cost, eval }) => {
                    println!("{eval}\t{cost:?}");
                    writer.write_summary(
                        SystemTime::now(),
                        (episodes * epoch + episode - 1) as i64,
                        cost.summary(),
                    )?;
                    writer.get_mut().flush()?;
                },
                ArgminImprovement::Unchanged => {},
            };
            if episode % episodes == 0 {
                println!("==== EPISODE: {episode} ====");
                let sizes = optimizer.get_trees().first().unwrap().sizes().collect::<Vec<_>>();
                println!("sizes: {sizes:?}");

                let graph = optimizer.get_trees()[0].graphviz();
                std::fs::write("tree.png", graph).unwrap();
                let graph = optimizer.get_trees()[BATCH-1].graphviz();
                std::fs::write("tree2.png", graph).unwrap();
            }
        }
        
        let loss = optimizer.par_update_model();
        let summary = loss.summary();
        writer.write_summary(
            SystemTime::now(),
            (episodes * epoch) as i64,
            summary,
        )?;
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
                let num_prohibitions = state.prohibited_actions.len() / C;
                if num_prohibitions == 0 {
                    let num_prohibitions = rng.gen_range(0..E);
                    *state = init_state(&mut rng, num_prohibitions);
                } else {
                    let num_prohibitions = rng.gen_range(0..num_prohibitions);
                    let s = state.state.clone();
                    let p = edges.choose_multiple(&mut rng, num_prohibitions).flat_map(|i| {
                        (0..C).map(|c| c * E + *i)
                    }).collect();
                    *state = new_state(s, p);
                }
            } else {
                n.retain(|(_, c, _)| *c == c_root_star);
                // dbg!(n.len());
                let (p, _, _) = n.choose(&mut rng).unwrap();
                if let Some(p) = p {
                    p.actions_taken().for_each(|a| {
                        let a = space.action(a);
                        space.act(state, &a);
                    })
                }
            }
        };
        optimizer.par_reset_trees(modify_root);
    }
    Ok(())
}
