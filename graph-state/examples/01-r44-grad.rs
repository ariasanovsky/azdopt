use std::{io::{BufWriter, Write}, time::SystemTime};

use az_discrete_opt::{tensorboard::{tf_path, Summarize}, state::{prohibit::WithProhibitions, layers::Layers}, space::layered::Layered, log::ArgminData, nabla::{optimizer::NablaOptimizer, model::dfdx::ActionModel, tree::node::{SamplePattern, SearchPolicy, TransitionMetadata}}, path::{set::ActionSet, ActionPath}};
use chrono::Utc;
use dfdx::{tensor::AutoDevice, tensor_ops::{AdamConfig, WeightDecay}, nn::{modules::ReLU, builders::Linear}};
use eyre::Context;
use graph_state::{bitset::primitive::B32, ramsey_counts::{RamseyCounts, space::RamseySpaceNoEdgeRecolor}, simple_graph::bitset_graph::ColoredCompleteBitsetGraph};
use rand::seq::SliceRandom;
use tensorboard_writer::TensorboardWriter;

const N: usize = 16;
const E: usize = N * (N - 1) / 2;
const C: usize = 3;
const SIZES: [usize; C] = [3, 3, 3];

type RawState = RamseyCounts<N, E, C, B32>;
type RichState = WithProhibitions<RawState>;

const STACK: usize = 1;
type S = Layers<RichState, STACK>;
type P = ActionSet;

type RichSpace = RamseySpaceNoEdgeRecolor<B32, N, E, C>;
type Space = Layered<STACK, RichSpace>;

const ACTION: usize = E * C;
const STATE: usize = STACK * 3 * C * E;

const HIDDEN_1: usize = 1024;
const HIDDEN_2: usize = 1024;
const HIDDEN_3: usize = 1024;

type ModelH = (
    (Linear<STATE, HIDDEN_1>, ReLU),
    (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
    (Linear<HIDDEN_2, HIDDEN_3>, ReLU),
    // (Linear<HIDDEN_3, ACTION>, dfdx::nn::modules::Sigmoid),
    (Linear<HIDDEN_3, ACTION>, ReLU),
);

const BATCH: usize = 1;

fn main() -> eyre::Result<()> {
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
    let dist = rand::distributions::WeightedIndex::new([1., 1., 1.])?;
    let init_state = |num_prohibited_edges: usize| -> S {
        let mut rng = rand::thread_rng();
        let g = ColoredCompleteBitsetGraph::generate(&dist, &mut rng);
        let s = RamseyCounts::new(g, &SIZES);
        let p = edges.choose_multiple(&mut rng, num_prohibited_edges).flat_map(|i| {
            (0..C).map(|c| c * E + *i)
        });
        let s = WithProhibitions::new(s, p);
        Layers::new(s)
    };

    let cfg = AdamConfig {
        lr: 3e-3,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: Some(WeightDecay::L2(1e-5)),
    };
    let model: ActionModel<ModelH, BATCH, STATE, ACTION> = ActionModel::new(dev, cfg);
    // let model = TrivialModel;
    const RICH_SPACE: RichSpace = RamseySpaceNoEdgeRecolor::new(SIZES, [1., 1., 1.]);
    const SPACE: Space = Layered::new(RICH_SPACE);

    // let max_num_root_actions = 25;

    let root_action_pattern = SamplePattern {
        head: 20,
        body: 20,
        tail: 10,
    };

    let mut optimizer: NablaOptimizer<_, _, P> = NablaOptimizer::par_new(
        SPACE, 
        || {
            init_state(E - 8)
        },
        model,
        BATCH,
        root_action_pattern.clone(),
    );
    let mut argmin = optimizer.par_argmin().map(|(s, c, e)| (ArgminData::new(s, c.clone(), 0, 0), e)).unwrap();
    println!("{}", argmin.1);
    
    writer.write_summary(
        SystemTime::now(),
        0,
        argmin.0.cost().summary(),
    )?;
    writer.get_mut().flush()?;
    
    let epochs: usize = 250;
    let episodes: usize = 800;

    for epoch in 1..epochs {
        println!("==== EPOCH: {epoch} ====");
        for episode in 1..=episodes {
            if episode % 800 == 0 {
                println!("==== EPISODE: {episode} ====");
                let max_lengths = optimizer.get_trees().iter().max_by_key(|t| t.sizes().sum::<usize>()).unwrap().sizes().collect::<Vec<_>>();
                println!("max_lengths: {max_lengths:?}");
                let min_lengths = optimizer.get_trees().iter().min_by_key(|t| t.sizes().sum::<usize>()).unwrap().sizes().collect::<Vec<_>>();
                println!("min_lengths: {min_lengths:?}");
                // println!("lengths: {lengths:?}");
                // let lengths = optimizer.get_trees().last().unwrap().sizes().collect::<Vec<_>>();
                // println!("lengths: {lengths:?}");
            }

            let episode_action_pattern = |_depth: usize| {
                SamplePattern {
                    head: 10,
                    body: 5,
                    tail: 0,
                }
            };
            let search_policy = |depth: usize| {
                match depth {
                    0 if episode < root_action_pattern.len() => SearchPolicy::Cyclic,
                    _ => SearchPolicy::Rating(|metadata| -> f32 {
                        let TransitionMetadata {
                            n_s,
                            c_s,
                            n_sa,
                            g_theta_star_sa,
                        } = metadata;
                        let exploit = g_theta_star_sa / c_s;
                        let explore = ((1 + n_s) as f32).sqrt() / (1 + n_sa) as f32;
                        exploit + 0.1 * explore
                    }),
                }
            };
            // optimizer.par_roll_out_episodes(episode_action_pattern, search_policy);
            optimizer.par_roll_out_episodes(search_policy);
            let episode_argmin = optimizer.par_argmin().map(|(s, c, e)| (ArgminData::new(s, c.clone(), episode, epoch), e)).unwrap();
            if episode_argmin.1 < argmin.1 {
                argmin = episode_argmin;
                writer.write_summary(
                    SystemTime::now(),
                    (episodes * epoch + episode) as i64,
                    argmin.0.cost().summary(),
                )?;
                writer.get_mut().flush()?;
                if argmin.1 < 0.01 {
                    println!("{}", argmin.0.short_form());
                }    
                println!("{}", argmin.1);
            }
        }
        
        let action_weights = |n_sa| {
            (n_sa as f32).sqrt()
        };
        let label = |s: &S, p: Option<&P>| -> usize {
            let actions_previously_taken = s.back().prohibited_actions.len() / C;
            let actions_taken = p.map(|p| p.len()).unwrap_or(0);
            actions_previously_taken + actions_taken
        };
        let label_weights = |l: usize| -> f32 {
            const MIN_LABEL: usize = E - 31;
            (1 + l - MIN_LABEL).pow(2) as f32
        };
        let state_weights = |s: &S| -> f32 {
            let label = label(s, None);
            label_weights(label)
        };
        let loss = optimizer.par_update_model(action_weights, state_weights);
        let summary = loss.summary();
        writer.write_summary(
            SystemTime::now(),
            (episodes * epoch) as i64,
            summary,
        )?;
        let summary = argmin.0.cost().summary();
        writer.write_summary(
            SystemTime::now(),
            (episodes * epoch) as i64,
            summary,
        )?;
        writer.get_mut().flush()?;
        let label_sample = |l: usize| -> SamplePattern {
            const LABEL_SIZE: usize = 3 * BATCH / 128;
            SamplePattern {
                head: LABEL_SIZE / 2,
                body: LABEL_SIZE - LABEL_SIZE / 2 - LABEL_SIZE / 4,
                tail: LABEL_SIZE / 4,
            }
        };
        let reset_state = |state: &mut S| {
            let mut s = state.back().clone();
            s.prohibited_actions.clear();
            let num_prohibited = E - epoch.clamp(4, 31);
            let mut rng = rand::thread_rng();
            let p = edges.choose_multiple(&mut rng, num_prohibited).flat_map(|i| {
                (0..C).map(|c| c * E + *i)
            });
            s.prohibited_actions.extend(p);
            *state = Layers::new(s);
        };
        let init_state = || {
            let num_prohibited = E - epoch.clamp(4, 31);
            init_state(num_prohibited)
        };
        optimizer.par_select_next_roots(reset_state, init_state, root_action_pattern.clone(), label, label_sample);
        let root_argmin = optimizer.par_argmin().map(|(s, c, e)| (ArgminData::new(s, c.clone(), 0, epoch + 1), e)).unwrap();
        if root_argmin.1 < argmin.1 {
            argmin = root_argmin;
            writer.write_summary(
                SystemTime::now(),
                (episodes * epoch) as i64,
                argmin.0.cost().summary(),
            )?;
            writer.get_mut().flush()?;
            if argmin.1 < 0.01 {
                println!("{}", argmin.0.short_form());
            }
            println!("{}", argmin.1);
        }
    }
    Ok(())
}
