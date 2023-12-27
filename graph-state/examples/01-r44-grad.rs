use std::{io::{BufWriter, Write}, time::SystemTime};

use az_discrete_opt::{tensorboard::{tf_path, Summarize}, state::{prohibit::WithProhibitions, layers::Layers}, space::layered::Layered, log::ArgminData, nabla::{optimizer::NablaOptimizer, model::dfdx::ActionModel}, path::sequence::ActionSequence};
use chrono::Utc;
use dfdx::{tensor::AutoDevice, tensor_ops::{AdamConfig, WeightDecay}, nn::{modules::ReLU, builders::Linear}};
use eyre::Context;
use graph_state::{bitset::primitive::B32, ramsey_counts::{RamseyCounts, space::RichRamseySpace}, simple_graph::bitset_graph::ColoredCompleteBitsetGraph};
use tensorboard_writer::TensorboardWriter;

const N: usize = 16;
const E: usize = N * (N - 1) / 2;
const C: usize = 3;
const SIZES: [usize; C] = [3, 3, 3];

type RawState = RamseyCounts<N, E, C, B32>;
type RichState = WithProhibitions<RawState>;

const STACK: usize = 4;
type S = Layers<RichState, STACK>;
type P = ActionSequence;

type RichSpace = RichRamseySpace<B32, N, E, C>;
type Space = Layered<STACK, RichSpace>;

const ACTION: usize = E * C;
const STATE: usize = STACK * 3 * C * E;

const HIDDEN_1: usize = 128;
const HIDDEN_2: usize = 128;

type ModelH = (
    (Linear<STATE, HIDDEN_1>, ReLU),
    (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
    (Linear<HIDDEN_2, ACTION>, ReLU),
);

const BATCH: usize = 1024;

fn main() -> eyre::Result<()> {
    let out_dir = tf_path().join("01-r44-grad").join(Utc::now().to_rfc3339());
    dbg!(&out_dir);
    let out_file = out_dir.join("tfevents-losses");
    // create the directory if it doesn't exist
    std::fs::create_dir_all(&out_dir).wrap_err("failed to create output directory")?;
    let writer = BufWriter::new(std::fs::File::create(out_file)?);
    let mut writer = TensorboardWriter::new(writer);
    writer.write_file_version()?;

    let dev = AutoDevice::default();
    let dist = rand::distributions::WeightedIndex::new([1., 1., 1.])?;
    let init_state = || -> S {
        let mut rng = rand::thread_rng();
        let g = ColoredCompleteBitsetGraph::generate(&dist, &mut rng);
        let s = RamseyCounts::new(g, &SIZES);
        let s = WithProhibitions::new(s, core::iter::empty());
        Layers::new(s)
    };

    let cfg = AdamConfig {
        lr: 2e-2,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: Some(WeightDecay::L2(1e-6)),
    };
    let model: ActionModel<ModelH, BATCH, STATE, ACTION> = ActionModel::new(dev, cfg);
    const RICH_SPACE: RichSpace = RichRamseySpace::new(SIZES, [1., 1., 1.]);
    const SPACE: Space = Layered::new(RICH_SPACE);

    let max_num_root_actions = 15;

    let mut optimizer: NablaOptimizer<_, _, P> = NablaOptimizer::par_new(SPACE, init_state, model, BATCH, max_num_root_actions);
    let mut argmin = optimizer.par_argmin().map(|(s, c, e)| (ArgminData::new(s, c.clone(), 0, 0), e)).unwrap();
    println!("{}", argmin.0.short_form());
    
    writer.write_summary(
        SystemTime::now(),
        0,
        argmin.0.cost().summary(),
    )?;
    writer.get_mut().flush()?;
    
    let epochs: usize = 25;
    let episodes: usize = 800;

    for epoch in 1..epochs {
        for episode in 1..=episodes {
            if episode % 100 == 0 {
                println!("==== EPISODE: {episode} ====");
                let lengths = optimizer.get_trees()[0].sizes().collect::<Vec<_>>();
                println!("lengths: {lengths:?}");
            }

            let max_num_actions = 3;

            optimizer.par_roll_out_episode(max_num_actions);
            let episode_argmin = optimizer.par_argmin().map(|(s, c, e)| (ArgminData::new(s, c.clone(), episode, epoch), e)).unwrap();
            if episode_argmin.1 < argmin.1 {
                argmin = episode_argmin;
                writer.write_summary(
                    SystemTime::now(),
                    (episodes * epoch + episode) as i64,
                    argmin.0.cost().summary(),
                )?;
                writer.get_mut().flush()?;
                println!("{}", argmin.0.short_form());
            }
        }
        
        let weights = |n_sa| {
            if n_sa == 0 {
                0.05
            } else {
                (n_sa as f32).sqrt()
            }
        };
        let loss = optimizer.par_update_model(weights);
        let summary = loss.summary();
        writer.write_summary(
            SystemTime::now(),
            (episodes * (epoch + 1)) as i64,
            summary,
        )?;
        let summary = argmin.0.cost().summary();
        writer.write_summary(
            SystemTime::now(),
            (episodes * (epoch + 1)) as i64,
            summary,
        )?;
        writer.get_mut().flush()?;
        todo!("clear trees");
        if epoch % 4 == 0 {
            // todo!("pick the next roots");
        }
    }
    Ok(())
}
