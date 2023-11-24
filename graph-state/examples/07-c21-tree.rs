#![feature(slice_flatten)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]

use itertools::Itertools;
use num_traits::MulAddAssign;
use rand_distr::Distribution;
use rayon::prelude::*;

use std::{
    io::{Write, BufWriter},
    mem::MaybeUninit,
    path::{Path, PathBuf}, ops::DivAssign, time::SystemTime,
};

use az_discrete_opt::{
    int_min_tree::{state_data::UpperEstimateData, INTMinTree},
    log::ArgminData,
    path::{set::ActionSet, ActionPath},
    space::StateActionSpace,
    state::{cost::Cost, prohibit::WithProhibitions},
};
use dfdx::{optim::Adam, prelude::*};
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
const RAW_STATE: usize = (N - 1) * (N - 2) / 2 - 1;
const STATE: usize = RAW_STATE + ACTION;
type NodeVector = [f32; STATE];
type ActionVec = [f32; ACTION];

const BATCH: usize = 256;

const HIDDEN_1: usize = 256;
const HIDDEN_2: usize = 256;

// type Core = (
//     (Linear<STATE, HIDDEN_1>, ReLU),
//     (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
//     // Linear<HIDDEN_2, PREDICTION>,
// );

type Logits = (
    (Linear<STATE, HIDDEN_1>, ReLU),
    (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
    Linear<HIDDEN_2, ACTION>,
    // Softmax,
);

type Valuation = (
    (Linear<STATE, HIDDEN_1>, ReLU),
    (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
    Linear<HIDDEN_2, 1>,
    // Linear<HIDDEN_2, 3>,
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
    
    let epochs: usize = 250;
    let episodes: usize = 800;
    
    let dev = AutoDevice::default();
    // let mut core_model = dev.build_module::<Core, f32>();
    let mut logits_model = dev.build_module::<Logits, f32>();
    let mut value_model = dev.build_module::<Valuation, f32>();
    
    let logits_config = AdamConfig {
        lr: 2e-2,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: Some(WeightDecay::L2(1e-6)), // Some(WeightDecay::Decoupled(1e-6)),
    };
    let value_config = AdamConfig {
        lr: 1e-2,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: Some(WeightDecay::L2(1e-6)), // Some(WeightDecay::Decoupled(1e-6)),
    };
    // let mut core_optimizer = Adam::new(&core_model, config.clone());
    let mut logits_optimizer = Adam::new(&logits_model, logits_config.clone());
    let mut value_optimizer = Adam::new(&value_model, value_config);

    // gradients
    let mut logits_gradients = logits_model.alloc_grads();
    let mut value_gradients = value_model.alloc_grads();
    
    // we initialize tensors to 0 and fill them as needed, minimizing allocations
    // input to model
    let mut state_vector_tensor: Tensor<Rank2<BATCH, STATE>, f32, _> = dev.zeros();
    // prediction tensors
    // let mut last_core_layer_prediction: Tensor<Rank2<BATCH, HIDDEN_2>, f32, _> = dev.zeros();
    let mut predicted_probabilities_tensor: Tensor<Rank2<BATCH, ACTION>, f32, _> = dev.zeros();
    let mut predicted_values_tensor: Tensor<Rank2<BATCH, 1>, f32, _> = dev.zeros();
    // observation tensors
    let mut observed_probabilities_tensor: Tensor<Rank2<BATCH, ACTION>, f32, _> = dev.zeros();
    let mut observed_values_tensor: Tensor<Rank2<BATCH, 1>, f32, _> = dev.zeros();
    // prediction arrays
    let mut predicted_probabilities: [ActionVec; BATCH] = [[0.0; ACTION]; BATCH];
    let mut predicted_values: [[f32; 1]; BATCH] = [[0.0; 1]; BATCH];
    // observation arrays
    let mut observed_probabilities: [ActionVec; BATCH] = [[0.0; ACTION]; BATCH];
    let mut observed_values: [[f32; 1]; BATCH] = [[0.0; 1]; BATCH];
    
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
        // println!(
        //     "{u_sa} = {g_sa_sum} / {n_sa} + {c_puct} * {p_sa} * ({n_s}.sqrt() / {n_sa})",
        // );
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
        // if cost.matching.len() < 2 { panicked for the star graph ;)
        //     println!("cost = {:?}", cost);
        //     println!("s = {:?}", s);
        //     println!("tree = {:?}", s);
        //     // dbg!(&s, &cost, &tree);
        //     panic!();
        // }
        cost
    };
    let (
        mut s_0,
        mut c_t,
    ): (
        [S; BATCH],
        [C; BATCH],
    ) = {
        let mut s_0: [MaybeUninit<S>; BATCH] = MaybeUninit::uninit_array();
        let mut c_t: [MaybeUninit<C>; BATCH] = MaybeUninit::uninit_array();
        (&mut s_0, &mut c_t)
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, (s_t, c_t)| {
                    let s = random_state(rng);
                    let c = cost(&s);
                    s_t.write(s);
                    c_t.write(c);
                },
            );
        (
            unsafe { MaybeUninit::array_assume_init(s_0) },
            unsafe { MaybeUninit::array_assume_init(c_t) },
        )
    };
    
    let mut p_t: [P; BATCH] = core::array::from_fn(|_| P::new());
    let mut candidate_data: ArgminData<C> = (&s_0, &c_t).into_par_iter().min_by(|(_, a), (_, b)| {
        a.evaluate().partial_cmp(&b.evaluate()).unwrap()
    }).map(|(s, c)| {
        ArgminData::new(s, c.clone(), 0, 0)
    }).unwrap();
    let mut all_losses: Vec<(f32, f32)> = vec![];
    const ALPHA: [f64; ACTION] = [0.03; ACTION];

    for epoch in 0..epochs {
        println!("==== EPOCH: {epoch} ====");
        // set state vectors
        let mut v_t: [NodeVector; BATCH] = [[0.0; STATE]; BATCH];
        (&s_0, &mut v_t)
            .into_par_iter()
            .for_each(|(s, v)| Space::write_vec(s, v));
        state_vector_tensor.copy_from(v_t.flatten());
        predicted_probabilities_tensor = logits_model
            .forward(state_vector_tensor.clone())
            .softmax::<Axis<1>>();
        predicted_probabilities_tensor.copy_into(predicted_probabilities.flatten_mut());
        let mut trees: [Tree; BATCH] = {
            let mut trees: [MaybeUninit<Tree>; BATCH] = MaybeUninit::uninit_array();
            (&mut trees, &mut predicted_probabilities, &c_t, &s_0)
                .into_par_iter()
                .for_each_init(
                    || rand::thread_rng(),
                    |rng, (t, root_predictions, cost, root)| {
                        let dir = rand_distr::Dirichlet::new(&ALPHA).unwrap();
                        let sample = dir.sample(rng);
                        root_predictions.iter_mut().zip_eq(sample.into_iter()).for_each(|(pi_theta, dir)| {
                            pi_theta.mul_add_assign(3., dir as f32);
                            pi_theta.div_assign(4.);
                        });
                        t.write(Tree::new::<Space>(root_predictions, cost.evaluate(), root));
                    });
            unsafe { MaybeUninit::array_assume_init(trees) }
        };
        for episode in 1..=episodes {
            if episode % 100 == 0 {
                println!("==== EPISODE: {episode} ====");
            }
            let mut s_t = s_0.clone();
            // todo! (perf) init once and clear each episode
            let mut transitions: [_; BATCH] = core::array::from_fn(|_| vec![]);
            // todo! tuck away the MU?
            let ends: [_; BATCH] = {
                let mut ends: [_; BATCH] = MaybeUninit::uninit_array();
                (&mut trees, &mut transitions, &mut s_t, &mut p_t, &mut ends)
                    .into_par_iter()
                    .for_each(|(t, trans, s_t, p_t, end)| {
                        p_t.clear();
                        trans.clear();
                        end.write(t.simulate_once::<Space>(s_t, p_t, trans, &upper_estimate));
                    });
                unsafe { MaybeUninit::array_assume_init(ends) }
            };
            (&s_t, &mut c_t)
                .into_par_iter()
                .for_each(|(s_t, c_t)| {
                    *c_t = cost(s_t);
                });

            (&s_t, &mut v_t).into_par_iter().for_each(|(s_t, v_t)| {
                Space::write_vec(s_t, v_t);
            });

            let episode_argmin = (&s_t, &c_t).into_par_iter().min_by(|(_, a), (_, b)| {
                a.evaluate().partial_cmp(&b.evaluate()).unwrap()
            }).unwrap();
            // update the global argmin
            if episode_argmin.1.evaluate() < candidate_data.cost().evaluate() {
                candidate_data = ArgminData::new(episode_argmin.0, episode_argmin.1.clone(), episode, epoch);
                println!("new min = {}", candidate_data.cost().evaluate());
                println!("argmin  = {candidate_data:?}");

                let summ = SummaryBuilder::new()
                    .scalar("cost/cost", candidate_data.cost().evaluate())
                    .scalar("cost/lambda_1", candidate_data.cost().lambda_1 as _)
                    .scalar("cost/mu", candidate_data.cost().matching.len() as _)
                    .build();
                // Write summaries to file.
                writer.write_summary(SystemTime::now(), (episodes * epoch + episode) as i64, summ)?;
                writer.get_mut().flush()?;
            }
            // copy tree root state vectors into tensor
            state_vector_tensor.copy_from(v_t.flatten());
            // calculate predicted probabilities
            // copy predicted probabilities into to host
            predicted_probabilities_tensor = logits_model
                .forward(state_vector_tensor.clone())
                .softmax::<Axis<1>>();
            predicted_probabilities_tensor.copy_into(predicted_probabilities.flatten_mut());
            // calculate predicted values
            // copy predicted values into host
            predicted_values_tensor = value_model.forward(state_vector_tensor.clone());
            predicted_values_tensor.copy_into(predicted_values.flatten_mut());
            let mut nodes: [Option<_>; BATCH] = core::array::from_fn(|_| None);
            (&mut nodes, ends, &c_t, &s_t, &predicted_values, &predicted_probabilities, &mut transitions)
                .into_par_iter()
                .for_each(|(n, end, c_t, s_t, v, probs_t, trans)| {
                    *n = end.update_existing_nodes::<Space>(c_t, s_t, probs_t, v, trans);
                });
            // insert nodes into trees
            (&mut trees, nodes).into_par_iter().for_each(|(t, n)| {
                if let Some(n) = n {
                    t.insert_node_at_next_level(n);
                }
            });
        }

        (&trees, &mut observed_probabilities, &mut observed_values)
            .into_par_iter()
            .for_each(|(t, p, v)| {
                t.observe(p, v);
            });
        // println!("values: {:?}", observed_values);
        observed_probabilities_tensor.copy_from(observed_probabilities.flatten());
        observed_values_tensor.copy_from(observed_values.flatten());

        (&s_0, &mut v_t)
            .into_par_iter()
            .for_each(|(s, v)| Space::write_vec(s, v));
        
        // update probability predictions
        state_vector_tensor.copy_from(v_t.flatten());
        let predicted_logits_traced = logits_model.forward(state_vector_tensor.clone().traced(logits_gradients));
        let cross_entropy =
            cross_entropy_with_logits_loss(predicted_logits_traced, observed_probabilities_tensor.clone());
        let entropy = cross_entropy.array();
        logits_gradients = cross_entropy.backward();
        logits_optimizer.update(&mut logits_model, &logits_gradients)
            .expect("optimizer failed");
        logits_model.zero_grads(&mut logits_gradients);
        
        // update mean max gain prediction
        // todo! unnecessary copy?
        state_vector_tensor.copy_from(v_t.flatten());
        let predicted_values_traced = value_model.forward(state_vector_tensor.clone().traced(value_gradients));
        let value_loss = mse_loss(predicted_values_traced, observed_values_tensor.clone());
        let mse = value_loss.array();
        value_gradients = value_loss.backward();
        value_optimizer.update(&mut value_model, &value_gradients)
            .expect("optimizer failed");
        value_model.zero_grads(&mut value_gradients);
        
        all_losses.push((entropy, mse));

        let summ = SummaryBuilder::new()
            .scalar("loss/entropy", entropy)
            .scalar("loss/mse", mse)
            .build();
        // Write summaries to file.
        writer.write_summary(SystemTime::now(), epoch as i64, summ)?;
        writer.get_mut().flush()?;

        let summ = SummaryBuilder::new()
            .scalar("cost/cost", candidate_data.cost().evaluate())
            .scalar("cost/lambda_1", candidate_data.cost().lambda_1 as _)
            .scalar("cost/mu", candidate_data.cost().matching.len() as _)
            .build();
        // Write summaries to file.
        writer.write_summary(SystemTime::now(), (episodes * epoch + episodes) as i64, summ)?;
        writer.get_mut().flush()?;
        
        
        let select_node = |_i, nodes: Vec<(_, _)>| {
            nodes[0].clone()
        };
        if epoch % 4 != 3 {
            (&mut s_0, trees).into_par_iter().enumerate().for_each(|(i, (s, t))| {
                let nodes = t.into_unstable_sorted_nodes();
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
            s_0.par_iter_mut().for_each(|s| {
                let mut rng = rand::thread_rng();
                *s = random_state(&mut rng);
            })
        }
    }
    dbg!(&out_dir);
    Ok(())
}
