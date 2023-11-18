#![feature(slice_flatten)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]

use rayon::prelude::*;

use std::{path::{Path, PathBuf}, io::Write, mem::MaybeUninit};

use az_discrete_opt::{int_min_tree::{INTMinTree, simulate_once::INTTransitions, state_data::UpperEstimateData}, log::{SimpleRootLog, ShortRootData}, path::{set::ActionSet, ActionPath}, state::{prohibit::WithProhibitions, cost::Cost}, tree_node::MutRefNode, space::{ActionSpace, StateActionSpace, StateSpaceVec, StateSpace}};
use dfdx::{optim::Adam, prelude::*};
use graph_state::simple_graph::{
    connected_bitset_graph::Conjecture2Dot1Cost,
    tree::{PrueferCode, space::modify_each_entry_once::ModifyEachPrueferCodeEntriesExactlyOnce},
};
use rand::rngs::ThreadRng;

use chrono::prelude::*;

use eyre::WrapErr;

const N: usize = 20;
type Space = ModifyEachPrueferCodeEntriesExactlyOnce<N>;

type RawState = PrueferCode<N>;
type S = WithProhibitions<RawState>;
type P = ActionSet;
type Node<'a> = MutRefNode<'a, S, P>;

type Tree = INTMinTree<P>;
type Trans<'a> = INTTransitions<'a, P>;
type C = Conjecture2Dot1Cost;
type Log = SimpleRootLog<S, C>;

const DEBUG_FALSE: bool = false;

const ACTION: usize = N * (N - 2);
const RAW_STATE: usize = N * (N - 2);
const STATE: usize = RAW_STATE + ACTION;
type NodeVector = [f32; STATE];
type ActionVec = [f32; ACTION];

const BATCH: usize = 1;

const HIDDEN_1: usize = 64;
const HIDDEN_2: usize = 64;

type Core = (
    (Linear<STATE, HIDDEN_1>, ReLU),
    (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
    // Linear<HIDDEN_2, PREDICTION>,
);

type Logits = (
    Linear<HIDDEN_2, ACTION>,
    // Softmax,
);

type Valuation = (
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
    let episodes: usize = 800;
    let max_before_resetting_actions = 5;
    let max_before_resetting_states = 10;

    let dev = AutoDevice::default();
    let mut core_model = dev.build_module::<Core, f32>();
    let mut logits_model = dev.build_module::<Logits, f32>();
    let mut value_model = dev.build_module::<Valuation, f32>();
    let mut opt = Adam::new(
        &core_model,
        AdamConfig {
            lr: 1e-3,
            betas: [0.9, 0.999],
            eps: 1e-8,
            weight_decay: Some(WeightDecay::L2(100.)),// Some(WeightDecay::Decoupled(1e-6)),
        },
    );

    // we initialize tensors to 0 and fill them as needed, minimizing allocations
    let mut v_t_tensor: Tensor<Rank2<BATCH, STATE>, f32, _> = dev.zeros();
    let mut prediction_tensor: Tensor<Rank2<BATCH, HIDDEN_2>, f32, _> = dev.zeros();
    let mut probs_tensor: Tensor<Rank2<BATCH, ACTION>, f32, _> = dev.zeros();
    let mut observed_probabilities_tensor: Tensor<Rank2<BATCH, ACTION>, f32, _> = dev.zeros();
    let mut observed_values_tensor: Tensor<Rank2<BATCH, 1>, f32, _> = dev.zeros();

    // generate states
    let random_state = |rng: &mut ThreadRng| {
        loop {
            let code = PrueferCode::generate(rng);
            let prohibited_actions = code.entries().map(|e| e.index::<Space>());
            let state = WithProhibitions::new(code.clone(), prohibited_actions);
            if !state.is_terminal::<Space>() {
                break state;
            }
        }
    };
    let mut s_0: [S; BATCH] =
        par_init_map(|| rand::thread_rng(), random_state);

    // calculate costs
    let cost = |s: &S| {
        type T = graph_state::simple_graph::tree::Tree<N>;
        let tree = T::from(&s.state);
        tree.conjecture_2_1_cost()
    };

    let upper_estimate = |estimate: UpperEstimateData| {
        let UpperEstimateData { n_s, n_sa, g_sa_sum, p_sa, depth } = estimate;
        debug_assert_ne!(n_sa, 0);
        let n_s = n_s as f32;
        let n_sa = n_sa as f32;
        let c_puct = 1.0;
        let g_sa = g_sa_sum / n_sa;
        let u_sa = g_sa + c_puct * p_sa * (n_s.sqrt() / n_sa);
        // println!(
        //     "{u_sa} = {g_sa_sum} / {n_sa} + {c_puct} * {p_sa} * ({n_s}.sqrt() / {n_sa})",
        // );
        u_sa
    };

    let mut episode_c_t: [C; BATCH] = core::array::from_fn(|_| Default::default());
    (&s_0, &mut episode_c_t).into_par_iter().for_each(|(s_t, c_t)| {
        let Conjecture2Dot1Cost {
            matching: m_t,
            lambda_1: l_1_t,
        } = c_t;
        let Conjecture2Dot1Cost { matching, lambda_1 } = cost(s_t);
        *m_t = matching;
        *l_1_t = lambda_1;
    });
    

    // set logs
    let mut logs: [Log; BATCH] = {
        let mut logs: [MaybeUninit<Log>; BATCH] = MaybeUninit::uninit_array();
        (&mut logs, &s_0, &episode_c_t)
            .into_par_iter()
            .for_each(|(l, s, cost)| {
                l.write(Log::new(cost, s));
            });
        unsafe { MaybeUninit::array_assume_init(logs) }
    };
    //Log::par_new_logs(&s_0, &episode_c_t);

    let mut all_losses: Vec<(f32, f32)> = vec![];
    for epoch in 0..epochs {
        println!("==== EPOCH: {epoch} ====");
        // set costs
        // set state vectors
        let mut v_t: [NodeVector; BATCH] = [[0.0; STATE]; BATCH];
        (&s_0, &mut v_t).into_par_iter().for_each(|(s, v)| {
            s.write_vec::<Space>(v)
        });
        println!("s_0[0] = {:?}", s_0[0]);
        println!("c_0[0] = {:?}", episode_c_t[0]);
        println!("v_0[0] = {:?}", v_t[0]);
        v_t_tensor.copy_from(v_t.flatten());
        prediction_tensor = core_model.forward(v_t_tensor.clone());
        probs_tensor = logits_model.forward(prediction_tensor.clone()).softmax::<Axis<1>>();
        let probs: [ActionVec; BATCH] = probs_tensor.array();
        println!("probs: {:?}", probs);
        let mut trees: [Tree; BATCH] = {
            let mut trees: [MaybeUninit<Tree>; BATCH] = MaybeUninit::uninit_array();
            (&mut trees, &probs, &episode_c_t, &s_0)
                .into_par_iter()
                .for_each(|(t, root_predictions, cost, root)| {
                    t.write(Tree::new::<Space>(root_predictions, cost.cost(), root));
                });
            // Tree::par_new_trees::<Space, BATCH, ACTION, _>(&probs, &episode_c_t, &s_0);
            unsafe { MaybeUninit::array_assume_init(trees) }
        };
        let mut grads = core_model.alloc_grads();
        for episode in 1..=episodes {
            if episode % 100 == 0 {
                println!("==== EPISODE: {episode} ====");
            }
            let mut s_t = s_0.clone();
            // todo! (perf) init once and clear each episode
            let mut p_t: [P; BATCH] = core::array::from_fn(|_| P::new());
            // todo! tuck away the MU?
            let transitions: [Trans; BATCH] = {
                let mut transitions: [MaybeUninit<Trans>; BATCH] = MaybeUninit::uninit_array();
                (&mut trees, &mut transitions, &mut s_t, &mut p_t).into_par_iter().for_each(|(t, trans, s, p)| {
                    let mut n_0 = MutRefNode::new(s, p);
                    trans.write(t.simulate_once::<Space>(&mut n_0, &upper_estimate));
                });
                // todo!();
                unsafe { MaybeUninit::array_assume_init(transitions) }
            };// Tree::par_simulate_once(&mut trees, &mut n_t);
            (&s_t, &mut episode_c_t).into_par_iter().for_each(|(s_t, c_t)| {
                *c_t = cost(s_t);
            });

            (&s_t, &mut v_t).into_par_iter().for_each(|(s_t, v_t)| {
                s_t.write_vec::<Space>(v_t);
            });

            (&mut logs, &s_t, &episode_c_t)
                .into_par_iter()
                .for_each(|(l, s_t, c_t)| {
                    l.update(s_t, c_t);
                });

            v_t_tensor.copy_from(v_t.flatten());
            prediction_tensor = core_model.forward(v_t_tensor.clone());
            probs_tensor = logits_model.forward(prediction_tensor.clone()).softmax::<Axis<1>>();
            let probs = probs_tensor.array();
            let values = value_model.forward(prediction_tensor.clone()).array();

            let mut nodes: [Option<_>; BATCH] =
                core::array::from_fn(|_| None);
            (&mut nodes, transitions, &episode_c_t, &s_t, &p_t, &values, &probs)
                .into_par_iter()
                .for_each(|(n, trans, c_t, s_t, p_t, v, probs_t)| {
                    // let c = m.len() as f32 + *l as f32;
                    *n = trans.update_existing_nodes::<Node, Space>(c_t, s_t, p_t, probs_t, v);
                });
            // insert nodes into trees
            (&mut trees, nodes).into_par_iter().for_each(|(t, n)| {
                if let Some(n) = n {
                    t.insert_node_at_next_level(n);
                }
            });
        }

        // trees[0].print_counts();
        // panic!();
        let mut probs: [ActionVec; BATCH] = [[0.0; ACTION]; BATCH];
        let mut values: [[f32; 1]; BATCH] = [[0.0; 1]; BATCH];

        (&trees, &mut probs, &mut values)
            .into_par_iter()
            .for_each(|(t, p, v)| {
                t.observe(p, v);
            });
        println!("values: {:?}", values);
        observed_probabilities_tensor.copy_from(probs.flatten());
        observed_values_tensor.copy_from(values.flatten());

        (&s_0, &mut v_t).into_par_iter().for_each(|(s, v)| {
            s.write_vec::<Space>(v)
        });
        let root_tensor = dev.tensor(v_t);
        let traced_predictions = core_model.forward(root_tensor.trace(grads));
        let predicted_logits = logits_model.forward(traced_predictions);
        let cross_entropy =
            cross_entropy_with_logits_loss(predicted_logits, observed_probabilities_tensor.clone());
        let entropy = cross_entropy.array();
        grads = cross_entropy.backward();

        let root_tensor = dev.tensor(v_t);
        let traced_predictions = core_model.forward(root_tensor.trace(grads));
        let predicted_values = value_model.forward(traced_predictions);
        let value_loss = mse_loss(predicted_values, observed_values_tensor.clone());
        let mse = value_loss.array();
        grads = value_loss.backward();

        all_losses.push((entropy, mse));
        dbg!(&all_losses);
        opt.update(&mut core_model, &grads)
            .expect("optimizer failed");
        core_model.zero_grads(&mut grads);

        let mut short_data: [Vec<ShortRootData<C>>; BATCH] = core::array::from_fn(|_| vec![]);

        (&mut logs, &mut s_0, &mut episode_c_t, &mut short_data)
            .into_par_iter()
            .for_each_init(
                || rand::thread_rng(),
                |rng, (log, s_0, c_t, d)| {
                    /* at the end of an epoch:
                     * 1. check for stagnation
                         * a. if stagnated, randomize root
                         * b. else, reset to a random time
                     * 2.
                     */
                    match log.stagnation() {
                        Some(stag) if stag > max_before_resetting_states /* && DEBUG_FALSE */ => {
                            // todo! s.reset_with(...);
                            // todo! stop resetting `s`'s prohibited_actions so we can train on roots with prohibited edges
                            // todo! or perhaps have two thresholds -- one for resetting the time randomly, one later for resetting the prohibited edges
                            let r = random_state(rng);
                            s_0.clone_from(&r);
                            log.reset_root(s_0, &cost(&s_0));
                            log.zero_stagnation();
                        },
                        Some(stag) if stag > max_before_resetting_actions /* && DEBUG_FALSE */ => {
                            // todo! s.reset_with(...);
                            // todo! stop resetting `s`'s prohibited_actions so we can train on roots with prohibited edges
                            // todo! or perhaps have two thresholds -- one for resetting the time randomly, one later for resetting the prohibited edges
                            s_0.prohibited_actions = s_0.state.entries().map(|e| e.index::<Space>()).collect();
                            log.reset_root(s_0, &cost(&s_0));
                        },
                        Some(_) => {
                            // todo! should `increment_stagnation` be private?
                            // maybe an `empty epoch` method is needed?
                            // return a `Vec` of serializable data to write to file instead of storing?
                            log.increment_stagnation();
                            // let t = rng.gen_range(1..=max_time);
                            // log.next_root_mut().reset(t);
                            // s_0.reset(t);
                        },
                        None => {
                            log.zero_stagnation();
                            // let t = rng.gen_range(1..=max_time);
                            // log.next_root_mut().reset(t);
                            s_0.clone_from(log.next_root());
                        },
                    }
                    c_t.clone_from(log.root_cost());
                    log.empty_root_data(d);
                },
            );
        // write {short_data:?} to out_dir/epoch{epoch}.json
        let epoch_path = out_dir.join(format!("epoch{epoch}")).with_extension("json");
        let mut epoch_file =
            std::fs::File::create(epoch_path).wrap_err("failed to create epoch file")?;
        // we print format!("{short_data:?}") to the file
        // we are not using serde lol
        epoch_file
            .write_all(format!("{short_data:?}").as_bytes())
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
