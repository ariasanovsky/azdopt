#![feature(slice_flatten)]
#![feature(maybe_uninit_uninit_array)]
#![feature(maybe_uninit_array_assume_init)]

use core::mem::MaybeUninit;
use core::ops::RangeInclusive;
use std::array::from_fn;

use rayon::prelude::*;

use az_discrete_opt::{state::{StateNode, StateVec}, int_min_tree::{INTMinTree, INTTransitions}, log::CostLog};
use dfdx::{prelude::*, optim::Adam};
use graph_state::simple_graph::connected_bitset_graph::ConnectedBitsetGraph;
use rand::{rngs::ThreadRng, Rng};

const N: usize = 31;
const E: usize = N * (N - 1) / 2;
type State = ConnectedBitsetGraph<N>;
type Node = StateNode<State>;

const ACTION: usize = 2 * E;
const STATE: usize = E + ACTION + 1;
type StateVector = [f32; STATE];
type ActionVec = [f32; ACTION];

const BATCH: usize = 1;

const HIDDEN_1: usize = 256;
const HIDDEN_2: usize = 128;

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

type Tree = INTMinTree;
type Trans = INTTransitions;


fn main() {
    let epochs: usize = 100;
    let episodes: usize = 1_000;

    let dev = AutoDevice::default();
    let mut core_model = dev.build_module::<Core, f32>();
    let mut logits_model = dev.build_module::<Logits, f32>();
    let mut value_model = dev.build_module::<Valuation, f32>();
    let mut opt = Adam::new(
        &core_model,
        AdamConfig {
            lr: 1e-2,
            betas: [0.5, 0.25],
            eps: 1e-6,
            weight_decay: Some(WeightDecay::Decoupled(1e-2)),
        },
    );
    
    // we initialize tensors to 0 and fill them as needed, minimizing allocations
    let mut v_t_tensor: Tensor<Rank2<BATCH, STATE>, f32, _> = dev.zeros();
    let mut prediction_tensor: Tensor<Rank2<BATCH, HIDDEN_2>, f32, _> = dev.zeros();
    let mut probs_tensor: Tensor<Rank2<BATCH, ACTION>, f32, _> = dev.zeros();
    let mut observed_probabilities_tensor: Tensor<Rank2<BATCH, ACTION>, f32, _> = dev.zeros();
    let mut observed_values_tensor: Tensor<Rank2<BATCH, 1>, f32, _> = dev.zeros();

    let mut times: RangeInclusive<usize> = 1usize..=5;
    let densities: RangeInclusive<f64> = 0.1f64..=0.9;
    
    let random_time = |rng: &mut ThreadRng| {
        rng.gen_range(times.clone())
    };
    let random_density = |rng: &mut ThreadRng| {
        rng.gen_range(densities.clone())
    };
    let random_connected_graph_state_node = |rng: &mut ThreadRng| {
        let time = random_time(rng);
        let density = random_density(rng);
        let graph = State::generate(density, rng);
        Node::new(graph, time)
    };

    // generate states
    let mut s_0: [Node; BATCH] = from_init(
        || rand::thread_rng(),
        random_connected_graph_state_node,
    );

    let cost = |s: &Node| s.state().conjecture_2_1_cost();
    let mut all_losses: Vec<(f32, f32)> = vec![];

    (1..=epochs).for_each(|epoch| {
        // set costs
        let mut c_t: [f32; BATCH] = [0.0; BATCH];
        (&s_0, &mut c_t).into_par_iter().for_each(|(s, c)| {
            *c = cost(s);
        });
        // set state vectors
        let mut v_t: [StateVector; BATCH] = [[0.0; STATE]; BATCH];
        (&s_0, &mut v_t).into_par_iter().for_each(|(s, v)| {
            s.write_vec(v);
        });
        // set logs
        let mut logs: [CostLog<Node>; BATCH] = CostLog::<Node>::par_new_logs(&s_0, &c_t);

        v_t_tensor.copy_from(&v_t.flatten());
        prediction_tensor = core_model.forward(v_t_tensor.clone());
        probs_tensor = logits_model.forward(prediction_tensor.clone());
        let probs: [ActionVec; BATCH] = probs_tensor.array();
        let mut trees: [Tree; BATCH] = Tree::par_new_trees(&probs, &c_t, &s_0);

        let mut grads = core_model.alloc_grads();
        (1..=episodes).for_each(|episode| {
            if episode % 100 == 0 {
                println!("==== EPISODE: {episode} ====");
            }
            let mut s_t = s_0.clone();
            let transitions: [Trans; BATCH] = Tree::par_simulate_once(&trees, &mut s_t);
            (&s_t, &mut c_t).into_par_iter().for_each(|(s, c)| {
                *c = s.state().conjecture_2_1_cost();
            });

            (&s_t, &mut v_t).into_par_iter().for_each(|(s, v)| {
                s.write_vec(v);
            });

            (&mut logs, &s_t, &c_t).into_par_iter().for_each(|(l, s, c)| {
                l.update(s, *c);
            });

            v_t_tensor.copy_from(v_t.flatten());
            prediction_tensor = core_model.forward(v_t_tensor.clone());
            probs_tensor = logits_model.forward(prediction_tensor.clone());
            let probs = probs_tensor.array();
            let values = value_model.forward(prediction_tensor.clone()).array();

            (&mut trees, &transitions, &c_t, &values).into_par_iter().for_each(|(t, trans, c, v)| {
                t.update(trans, *c, v);
            });

            (&mut trees, &transitions, &probs, &c_t, &s_t).into_par_iter().for_each(|(t, trans, p, c, s)| {
                t.insert(trans, s, *c, p);
            });
        });

        let mut probs: [ActionVec; BATCH] = [[0.0; ACTION]; BATCH];
        let mut values: [[f32; 1]; BATCH] = [[0.0; 1]; BATCH];

        (&trees, &mut probs, &mut values).into_par_iter().for_each(|(t, p, v)| {
            t.observe(p, v);
        });
        observed_probabilities_tensor.copy_from(probs.flatten());
        observed_values_tensor.copy_from(values.flatten());

        (&s_0, &mut v_t).into_par_iter().for_each(|(s, v)| {
            s.write_vec(v);
        });
        let root_tensor = dev.tensor(v_t);
        let traced_predictions = core_model.forward(root_tensor.trace(grads));
        let predicted_logits = logits_model.forward(traced_predictions);
        let cross_entropy = cross_entropy_with_logits_loss(predicted_logits, observed_probabilities_tensor.clone());
        let entropy = cross_entropy.array();
        grads = cross_entropy.backward();

        let root_tensor = dev.tensor(v_t);
        let traced_predictions = core_model.forward(root_tensor.trace(grads));
        let predicted_values = value_model.forward(traced_predictions);
        let value_loss = mse_loss(predicted_values, observed_values_tensor.clone());
        let mse = value_loss.array();
        grads = value_loss.backward();

        all_losses.push((entropy, mse));
        opt.update(&mut core_model, &grads).expect("optimizer failed");
        core_model.zero_grads(&mut grads);

        times = *times.start()..=(times.end() + 1);
        todo!("more updates required, also include a checker for stagnation")
    })
}

fn from_init<const N: usize, R, T: Send>(
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