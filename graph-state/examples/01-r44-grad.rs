use std::io::BufWriter;

use az_discrete_opt::{tensorboard::tf_path, state::{prohibit::WithProhibitions, layers::Layers}, space::layered::Layered, log::ArgminData, nabla::{optimizer::NablaOptimizer, model::dfdx::ActionModel}, path::set::ActionSet};
use chrono::Utc;
use dfdx::{tensor::AutoDevice, tensor_ops::{AdamConfig, WeightDecay}, nn::{modules::ReLU, builders::Linear}};
use eyre::Context;
use graph_state::{bitset::primitive::B32, ramsey_counts::{RamseyCounts, space::RichRamseySpace, TotalCounts}, simple_graph::bitset_graph::ColoredCompleteBitsetGraph};
use tensorboard_writer::TensorboardWriter;

const N: usize = 17;
const E: usize = N * (N - 1) / 2;

type RawState = RamseyCounts<N, E, 2, B32>;
type RichState = WithProhibitions<RawState>;

const STACK: usize = 8;
type S = Layers<RichState, STACK>;
type C = TotalCounts<2>;
type P = ActionSet;

type RichSpace = RichRamseySpace<B32, N, E>;
type Space = Layered<STACK, RichSpace>;

const ACTION: usize = E * 2;
const STATE: usize = 8 * 6 * E;

const HIDDEN_1: usize = 256;
const HIDDEN_2: usize = 256;

type ModelH = (
    (Linear<STATE, HIDDEN_1>, ReLU),
    (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
    Linear<HIDDEN_2, ACTION>,
);

const BATCH: usize = 4096;

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
    let h_config = AdamConfig {
        lr: 2e-2,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: Some(WeightDecay::L2(1e-6)),
    };

    let dist = rand::distributions::WeightedIndex::new([1., 1.]).unwrap();
    let init_state = || -> S {
        let mut rng = rand::thread_rng();
        const SIZES: [usize; 2] = [4, 4];
        let g = ColoredCompleteBitsetGraph::generate(&dist, &mut rng);
        let s = RamseyCounts::new(g, &SIZES);
        let s = WithProhibitions::new(s, core::iter::empty());
        Layers::new(s)
    };

    let model: ActionModel<ModelH, BATCH, STATE, ACTION> = ActionModel::new(dev);
    let evaluate = |c: &C| -> f32 {
        c.0.iter().sum::<i32>() as f32
    };
    let space: RichSpace = RichRamseySpace::new(evaluate);
    let space: Space = Layered::new(space);

    let mut optimizer: NablaOptimizer<_, _, P> = NablaOptimizer::par_new(space, init_state, model, BATCH);
    let (s, c, e) = optimizer.argmin().unwrap();
    let mut argmin = ArgminData::new(s, c, 0, 0);

    let epochs: usize = 25;
    let episodes: usize = 800;

    for epoch in 0..epochs {
        for episode in 1..=episodes {
            if episode % 100 == 0 {
                println!("==== EPISODE: {episode} ====");
            }
            optimizer.par_roll_out_episode();
            todo!("update the argmin");
            todo!("write summary to tensorboard");
        }
        
        let weights = |n_sa: usize| {
            if n_sa == 0 {
                0.05
            } else {
                (n_sa as f32).sqrt()
            }
        };
        let loss = optimizer.update_model(weights);
        todo!("write loss");
        todo!("write best cost");
        todo!("pick the next roots");
    }
    Ok(())
}
