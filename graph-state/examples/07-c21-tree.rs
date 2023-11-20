#![feature(slice_flatten)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]

use rayon::prelude::*;

use std::{
    io::Write,
    mem::MaybeUninit,
    path::{Path, PathBuf},
};

use az_discrete_opt::{
    int_min_tree::{state_data::UpperEstimateData, INTMinTree},
    log::{RootCandidateData, NextEpochRoot},
    path::{set::ActionSet, ActionPath},
    space::{ActionSpace, StateSpace, StateSpaceVec},
    state::{cost::Cost, prohibit::WithProhibitions},
};
use dfdx::{optim::Adam, prelude::*};
use graph_state::simple_graph::{
    connected_bitset_graph::Conjecture2Dot1Cost,
    tree::{space::modify_each_entry_once::ModifyEachPrueferCodeEntriesExactlyOnce, PrueferCode},
};
use rand::rngs::ThreadRng;

use chrono::prelude::*;

use eyre::WrapErr;

const N: usize = 20;
type Space = ModifyEachPrueferCodeEntriesExactlyOnce<N>;

type RawState = PrueferCode<N>;
type S = WithProhibitions<RawState>;
type P = ActionSet;
// type Node<'a> = MutRefNode<'a, S, P>;

type Tree = INTMinTree<P>;
type C = Conjecture2Dot1Cost;
// type Log = SimpleRootLog<S, C>;

const ACTION: usize = N * (N - 2);
const RAW_STATE: usize = N * (N - 2);
const STATE: usize = RAW_STATE + ACTION;
type NodeVector = [f32; STATE];
type ActionVec = [f32; ACTION];

const BATCH: usize = 64;

const HIDDEN_1: usize = 64;
const HIDDEN_2: usize = 64;

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

fn main() -> eyre::Result<()> {
    // `epoch0.json, ...` get written to OUT/{dir_name}
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
    // create the directory if it doesn't exist
    std::fs::create_dir_all(&out_dir).wrap_err("failed to create output directory")?;

    let epochs: usize = 250;
    let episodes: usize = 100;
    let max_before_resetting_actions = 5;
    let max_before_resetting_states = 10;

    let dev = AutoDevice::default();
    // let mut core_model = dev.build_module::<Core, f32>();
    let mut logits_model = dev.build_module::<Logits, f32>();
    let mut value_model = dev.build_module::<Valuation, f32>();
    
    let logits_config = AdamConfig {
        lr: 0.02,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: Some(WeightDecay::L2(1e-6)), // Some(WeightDecay::Decoupled(1e-6)),
    };
    let value_config = AdamConfig {
        lr: 0.002,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: None, //Some(WeightDecay::L2(1e-6)), // Some(WeightDecay::Decoupled(1e-6)),
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
        let c_puct = 10.0;
        let g_sa = g_sa_sum / n_sa;
        let u_sa = g_sa + c_puct * p_sa * (n_s.sqrt() / n_sa);
        // println!(
        //     "{u_sa} = {g_sa_sum} / {n_sa} + {c_puct} * {p_sa} * ({n_s}.sqrt() / {n_sa})",
        // );
        u_sa
    };
    
    // generate states
    let default_prohibitions = |s: &RawState| {
        s.entries().map(|e| e.index::<Space>()).collect::<Vec<_>>()
    };
    let random_state = |rng: &mut ThreadRng| loop {
        let code = PrueferCode::generate(rng);
        let prohibited_actions = default_prohibitions(&code);
        let state = WithProhibitions::new(code.clone(), prohibited_actions);
        if !state.is_terminal::<Space>() {
            break state;
        }
    };
    // let mut s_0: [S; BATCH] = par_init_map(|| rand::thread_rng(), random_state);
    // calculate costs
    let cost = |s: &S| {
        type T = graph_state::simple_graph::tree::Tree<N>;
        let tree = T::from(&s.state);
        let cost = tree.conjecture_2_1_cost();
        if cost.cost() < 100.0 {
            println!("\tcost = {cost:?}");
            println!("\tevaluates to: {}", cost.cost());
            println!("\ts = {s:?}");
        }
        cost
    };
    let (
        mut s_0,
        mut c_t,
        mut next_root,
    ): (
        [S; BATCH],
        [C; BATCH],
        [NextEpochRoot<S, C>; BATCH],
    ) = {
        let mut s_0: [MaybeUninit<S>; BATCH] = MaybeUninit::uninit_array();
        let mut c_t: [MaybeUninit<C>; BATCH] = MaybeUninit::uninit_array();
        let mut next_root: [MaybeUninit<NextEpochRoot<S, C>>; BATCH] = MaybeUninit::uninit_array();
        (&mut s_0, &mut c_t, &mut next_root)
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, (s_t, c_t, next_root)| {
                    let s = random_state(rng);
                    let c = cost(&s);
                    next_root.write(NextEpochRoot::new(s.clone(), c.clone()));
                    s_t.write(s);
                    c_t.write(c);
                },
            );
        (
            unsafe { MaybeUninit::array_assume_init(s_0) },
            unsafe { MaybeUninit::array_assume_init(c_t) },
            unsafe { MaybeUninit::array_assume_init(next_root) },
        )
    };
    
    let mut argmin =
        next_root
        .par_iter()
        .map(|log| log.current_candidate())
        .max_by(|a, b| {
            a.cost().cost().partial_cmp(&b.cost().cost()).unwrap()
        }).unwrap();

    let mut p_t: [P; BATCH] = core::array::from_fn(|_| P::new());
    let mut candidate_data: [Vec<RootCandidateData<C>>; BATCH] = core::array::from_fn(|_| vec![]);
    let mut all_losses: Vec<(f32, f32)> = vec![];
    for epoch in 0..epochs {
        println!("==== EPOCH: {epoch} ====");
        // set costs
        // set state vectors
        let mut v_t: [NodeVector; BATCH] = [[0.0; STATE]; BATCH];
        (&s_0, &mut v_t)
            .into_par_iter()
            .for_each(|(s, v)| s.write_vec::<Space>(v));
        // println!("s_0[0] = {:?}", s_0[0]);
        // println!("c_0[0] = {:?}", c_t[0]);
        // println!("v_0[0] = {:?}", v_t[0]);
        state_vector_tensor.copy_from(v_t.flatten());
        // last_core_layer_prediction = core_model.forward(state_vector_tensor.clone());
        predicted_probabilities_tensor = logits_model
            .forward(state_vector_tensor.clone())
            .softmax::<Axis<1>>();
        predicted_probabilities_tensor.copy_into(predicted_probabilities.flatten_mut());
        // println!("probs: {:?}", predicted_probabilities);
        let mut trees: [Tree; BATCH] = {
            let mut trees: [MaybeUninit<Tree>; BATCH] = MaybeUninit::uninit_array();
            (&mut trees, &predicted_probabilities, &c_t, &s_0)
                .into_par_iter()
                .for_each(|(t, root_predictions, cost, root)| {
                    t.write(Tree::new::<Space>(root_predictions, cost.cost(), root));
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
                s_t.write_vec::<Space>(v_t);
            });

            (&mut next_root, &mut candidate_data, &s_t, &c_t)
                .into_par_iter()
                .for_each(|(l, candidate_data, s_t, c_t)| {
                    if let Some(data) = l.post_episode_update(s_t, c_t) {
                        candidate_data.push(data);
                    }
                });
            // copy tree root state vectors into tensor
            // propagate through core model
            state_vector_tensor.copy_from(v_t.flatten());
            // last_core_layer_prediction = core_model.forward(state_vector_tensor.clone());
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
            .for_each(|(s, v)| s.write_vec::<Space>(v));
        
        // let (
        //     (
        //         a,
        //         c,
        //     ),
        //     (
        //         b,
        //         d,
        //     )
        // ) = &core_model;
        // let dfdx::nn::modules::Linear {
        //     weight,
        //     bias,
        // } = a;
        // let core_mean = weight.clone().mean::<_, Axis<0>>().mean::<_, Axis<0>>().array();
        // let (
        //     (a, _),
        //     (b, _),
        //     c,
        // ) = &logits_model;
        // let dfdx::nn::modules::Linear {
        //     weight,
        //     bias,
        // } = a;
        // let logits_mean = weight.clone().mean::<_, Axis<0>>().mean::<_, Axis<0>>().array();
        // let (
        //     (a, _),
        //     (b, _),
        //     c,
        // ) = &value_model;
        // let dfdx::nn::modules::Linear {
        //     weight,
        //     bias,
        // } = a;
        // let value_mean = weight.clone().mean::<_, Axis<0>>().mean::<_, Axis<0>>().array();
        // dbg!(logits_mean, value_mean);
        
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
        println!("all_losses: {all_losses:?}");

        (&mut next_root, &mut s_0, &mut c_t, &mut candidate_data)
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, (next_root, s_0, c_t, d)| {
                    todo!()
                }
            );
        todo!("
        let epoch_argmin = logs.par_iter().map(|log| log.short_data()).max_by(|a, b| {{
            a.cost().cost().partial_cmp(&b.cost().cost()).unwrap()
        }}).unwrap();
        if epoch_argmin.cost().cost() < argmin.cost().cost() {{
            argmin = epoch_argmin;
            println!(\"new min = {{}}\", argmin.cost().cost());
            println!(\"argmin  = {argmin:?}\");
        }}");
        // write {short_data:?} to out_dir/epoch{epoch}.json
        let epoch_path = out_dir.join(format!("epoch{epoch}")).with_extension("json");
        let mut epoch_file =
            std::fs::File::create(epoch_path).wrap_err("failed to create epoch file")?;
        // we print format!("{short_data:?}") to the file
        // we are not using serde lol
        epoch_file
            .write_all(format!("{candidate_data:?}").as_bytes())
            .wrap_err("failed to write epoch file")?;
    }
    dbg!(&out_dir);
    Ok(())
}

fn par_init_map<const N: usize, R, T: Send>(
    init: (impl Fn() -> R + Sync),
    f: impl Fn(&mut R) -> T + Sync,
) -> [T; N] {
    let mut t: [MaybeUninit<T>; N] = MaybeUninit::uninit_array();
    t.par_iter_mut().for_each(|t| {
        let mut r = init();
        t.write(f(&mut r));
    });
    unsafe { MaybeUninit::array_assume_init(t) }
}

// trait ParMap<'a>: IntoParallelIterator + 'a {
//     type Input;
//     type Output<U>;

//     fn par_map<U>(
//         self,
//         f: impl Fn(Self::Item) -> U + Sync,
//     ) -> Self::Output<U> where
//         U: Send,
//     ;

//     fn par_map_mut<U, M>(
//         self,
//         m: &mut Self::Output<M>,
//         f: impl Fn(Self::Input, &mut M) -> U + Sync,
//     ) -> Self::Output<U> where
//         U: Send,
//         M: Send,
//     ;
// }

// impl<'a, T: Sync + 'a, const N: usize> ParMap<'a> for &'a [T; N] {
//     type Input = &'a T;

//     type Output<U> = [U; N];

//     fn par_map<U>(
//         self,
//         f: impl Fn(Self::Input) -> U + Sync,
//     ) -> Self::Output<U> where
//         U: Send,
//     {
//         let mut u: [MaybeUninit<U>; N] = MaybeUninit::uninit_array();
//         (&mut u, self).into_par_iter().for_each(|(u, t)| {
//             u.write(f(t));
//         });
//         unsafe { MaybeUninit::array_assume_init(u) }
//     }

//     fn par_map_mut<U, M>(
//         self,
//         m: &mut Self::Output<M>,
//         f: impl Fn(Self::Input, &mut M) -> U + Sync,
//     ) -> Self::Output<U> where
//         U: Send,
//         M: Send,
//     {
//         let mut u: [MaybeUninit<U>; N] = MaybeUninit::uninit_array();
//         (&mut u, self, m).into_par_iter().for_each(|(u, t, m)| {
//             u.write(f(t, m));
//         });
//         unsafe { MaybeUninit::array_assume_init(u) }
//     }
// }

// impl<'a, S: Sync + 'a, T: Sync + 'a, const N: usize> ParMap<'a> for (&'a [S; N], &'a [T; N]) {
//     type Input = (&'a S, &'a T);

//     type Output<U> = [U; N];

//     fn par_map<U>(
//         self,
//         f: impl Fn(Self::Input) -> U + Sync,
//     ) -> Self::Output<U> where
//         U: Send,
//     {
//         let mut u: [MaybeUninit<U>; N] = MaybeUninit::uninit_array();
//         (&mut u, self.0, self.1).into_par_iter().for_each(|(u, s, t)| {
//             u.write(f((s, t)));
//         });
//         unsafe { MaybeUninit::array_assume_init(u) }
//     }

//     fn par_map_mut<U, M>(
//         self,
//         m: &mut Self::Output<M>,
//         f: impl Fn(Self::Input, &mut M) -> U + Sync,
//     ) -> Self::Output<U> where
//         U: Send,
//         M: Send,
//     {
//         let mut u: [MaybeUninit<U>; N] = MaybeUninit::uninit_array();
//         (&mut u, self.0, self.1, m).into_par_iter().for_each(|(u, s, t, m)| {
//             u.write(f((s, t), m));
//         });
//         unsafe { MaybeUninit::array_assume_init(u) }
//     }
// }

// impl<'a, T1: Sync + 'a, T2: Sync + 'a, T3: Sync + 'a, const N: usize> ParMap<'a> for (&'a [T1; N], &'a [T2; N], &'a [T3; N]) {
//     type Input = (&'a T1, &'a T2, &'a T3);

//     type Output<U> = [U; N];

//     fn par_map<U>(
//         self,
//         f: impl Fn(Self::Input) -> U + Sync,
//     ) -> Self::Output<U> where
//         U: Send,
//     {
//         let mut u: [MaybeUninit<U>; N] = MaybeUninit::uninit_array();
//         (&mut u, self.0, self.1, self.2).into_par_iter().for_each(|(u, t1, t2, t3)| {
//             u.write(f((t1, t2, t3)));
//         });
//         unsafe { MaybeUninit::array_assume_init(u) }
//     }

//     fn par_map_mut<U, M>(
//         self,
//         m: &mut Self::Output<M>,
//         f: impl Fn(Self::Input, &mut M) -> U + Sync,
//     ) -> Self::Output<U> where
//         U: Send,
//         M: Send,
//     {
//         let mut u: [MaybeUninit<U>; N] = MaybeUninit::uninit_array();
//         (&mut u, self.0, self.1, self.2, m).into_par_iter().for_each(|(u, t1, t2, t3, m)| {
//             u.write(f((t1, t2, t3), m));
//         });
//         unsafe { MaybeUninit::array_assume_init(u) }
//     }
// }

// impl<'a, T1: Sync + 'a, T2: Sync + 'a, T3: Sync + 'a, T4: Sync + 'a, const N: usize> ParMap<'a> for (&'a [T1; N], &'a [T2; N], &'a [T3; N], &'a [T4; N]) {
//     type Input = (&'a T1, &'a T2, &'a T3, &'a T4);

//     type Output<U> = [U; N];

//     fn par_map<U>(
//         self,
//         f: impl Fn(Self::Input) -> U + Sync,
//     ) -> Self::Output<U> where
//         U: Send,
//     {
//         let mut u: [MaybeUninit<U>; N] = MaybeUninit::uninit_array();
//         (&mut u, self.0, self.1, self.2, self.3).into_par_iter().for_each(|(u, t1, t2, t3, t4)| {
//             u.write(f((t1, t2, t3, t4)));
//         });
//         unsafe { MaybeUninit::array_assume_init(u) }
//     }

//     fn par_map_mut<U, M>(
//         self,
//         m: &mut Self::Output<M>,
//         f: impl Fn(Self::Input, &mut M) -> U + Sync,
//     ) -> Self::Output<U> where
//         U: Send,
//         M: Send,
//      {
//         let mut u: [MaybeUninit<U>; N] = MaybeUninit::uninit_array();
//         (&mut u, self.0, self.1, self.2, self.3, m).into_par_iter().for_each(|(u, t1, t2, t3, t4, m)| {
//             u.write(f((t1, t2, t3, t4), m));
//         });
//         unsafe { MaybeUninit::array_assume_init(u) }
//     }
// }
