use std::mem::MaybeUninit;

use azopt::ir_tree::ir_min_tree::{IRMinTree, IRState};
use dfdx::{
    optim::Adam,
    prelude::*,
    tensor::AutoDevice,
};
use ramsey::ColoredGraphWithCounts;
use rayon::prelude::{ParallelIterator, IntoParallelRefMutIterator, IndexedParallelIterator, IntoParallelRefIterator};

/* (S, A) is as follows:
    * each s in S is equivalent to the following triple (g, a, t) where:
        * g is a 2-edge-colored complete graph on N = 17 vertices
        * a is the set of recoloring actions that may be taken from s
        * t is the number of remaining recolorings before termination
    * each element in A is a pair (e, c) where
        * e is an edge of g
        * c is a color (red or blue)
    * taking action (e, c) from state s = (g, a, t) results in the state s' = (g', a', t') where:
        * g' is the graph obtained from g by recoloring e with c
        * t' = t - 1
        * all actions (e, c') are removed from p'; furthermore if t' = 0, then a' is emptied
            * i.e., the same edge may not be recolored, and no actions may be taken if t' = 0
    * to evaluate s, we convert it into a tensor product of the following vectors:
        * c_red, c_blue where the uv-th entry equals:
            * the number of K_4's in g_red (g_blue) + uv
        * e_red, e_blue where the uv-th entry equals:
            * 1.0 or 0.0 if the uv-th edge is red (blue)
        * a_red, a_blue where the uv-th entry equals:
            * 1.0 or 0.0 if the action to recolor the uv-th edge red (blue) may be taken
        * t (as a f32)
    * the cost of s is the total number of monochromatic K_4's in g
*/
const N: usize = 17;
const E: usize = (N * (N - 1)) / 2;
const STATE: usize = 6 * E + 1;
const ACTION: usize = 2 * E;

/* the model is a neural network f_theta: S -> R^{|A| + 1} = (p_theta, g_theta) where:
    * p_theta: S -> R^{|A|} is a policy vector
    * g_theta: S -> R is a scalar value
    // todo!()
    // * g_theta: S -> R^3 is a triple of scalar values
    //     * the components of g_theta correspond to:
    //         * g_{theta, max}, g_{theta, mean_max}, g_{theta, mean} (explained elsewhere)
    * to measure loss against an observation (p_obs, g_obs), our loss function has terms
        * value error: (g_{obs, *} - g_{theta, *})^2,
        * cross-entropy: -<p_obs, log(p_theta)>, and
        * regularization: ||theta||^2.
        * Loss is processed in batches of observations based on different states.
*/

// model constants
const HIDDEN_1: usize = 256;
const HIDDEN_2: usize = 128;
const PREDICTION: usize = ACTION + 1;

type Architecture = (
    (Linear<STATE, HIDDEN_1>, ReLU),
    (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
    Linear<HIDDEN_2, PREDICTION>,
);

// type Logits = (
//     Linear<HIDDEN_2, ACTION>,
//     // Softmax,
// );

// type Valuation = (
//     Linear<HIDDEN_2, 1>,
//     // Linear<HIDDEN_2, 3>,
// );

// search constants
const BATCH: usize = 64;

#[derive(Debug, Clone)]
struct GraphState {
    graph: ColoredGraphWithCounts,
    actions: [[bool; E]; 2],
    time: usize,
}

impl IRState for GraphState {
    fn cost(&self) -> f32 {
        let Self { graph, actions: _, time: _} = self;
        let ColoredGraphWithCounts { edges, neighborhoods: _, counts } = graph;
        edges.into_iter().zip(counts.into_iter()).map(|(edges, counts)| {
            edges.into_iter().zip(counts.into_iter()).filter_map(|(e, c)| {
                if *e {
                    Some(c)
                } else {
                    None
                }
            }).sum::<i32>()
        }).sum::<i32>() as f32
    }

    fn action_rewards(&self) -> Vec<(usize, f32)> {
        let Self { graph, actions: _, time: _} = self;
        let ColoredGraphWithCounts { edges, neighborhoods: _, counts } = graph;
        todo!()
    }
}

impl GraphState {
    fn generate_random<R: rand::Rng>(t: usize, rng: &mut R) -> Self {
        let graph = ColoredGraphWithCounts::generate_random(rng);
        let actions = graph.edges().map(|graph| {
            graph.map(|e| !e)
        });
        Self { graph, actions, time: t }
    }

    fn to_vec(&self) -> [f32; 6 * E + 1] {
        let mut vec = Vec::with_capacity(6 * E + 1);
        vec.extend(self.graph.edges().iter().flatten().map(|e| if *e { 1.0 } else { 0.0 }));
        vec.extend(self.graph.counts().iter().flatten().map(|c| *c as f32));
        vec.extend(self.actions.iter().flatten().map(|a| if *a { 1.0 } else { 0.0 }));
        vec.push(self.time as f32);
        vec.try_into().unwrap()
    }
}

type Tree = IRMinTree<GraphState>;
type StateVec = [f32; STATE];
type PredictionVec = [f32; PREDICTION];

fn main() {
    let dev = AutoDevice::default();
    let mut model: ((modules::Linear<817, 256, f32, Cuda>, ReLU), (modules::Linear<256, 128, f32, Cuda>, ReLU), modules::Linear<128, 273, f32, Cuda>) = dev.build_module::<Architecture, f32>();
    let mut opt = Adam::new(
        &model,
        AdamConfig {
            lr: 1e-2,
            betas: [0.5, 0.25],
            eps: 1e-6,
            weight_decay: Some(WeightDecay::Decoupled(1e-2)),
         }
    );

    let mut graphs: [MaybeUninit<GraphState>; BATCH] = unsafe {
        MaybeUninit::uninit().assume_init()
    };
    graphs.par_iter_mut().for_each(|g| {
        g.write(GraphState::generate_random(10, &mut rand::thread_rng()));
    });
    let mut graphs: [GraphState; BATCH] = unsafe {
        core::mem::transmute(graphs)
    };
    let mut states: [MaybeUninit<StateVec>; BATCH] = unsafe {
        MaybeUninit::uninit().assume_init()
    };
    states.par_iter_mut().zip_eq(graphs.par_iter()).for_each(|(s, g)| {
        s.write(g.to_vec());
    });
    let mut states: [StateVec; BATCH] = unsafe {
        core::mem::transmute(states)
    };
    let mut state_tensor = dev.tensor(states);
    let mut prediction_tensor = model.forward(state_tensor);
    let predictions: [PredictionVec; BATCH] = prediction_tensor.array();
    
    let mut trees: [MaybeUninit<Tree>; BATCH] = unsafe {
        core::mem::MaybeUninit::uninit().assume_init()
    };
    trees.par_iter_mut().zip_eq(graphs.par_iter().zip_eq(predictions.par_iter())).for_each(|(t, (g, p))| {
        let probs = &p[0..ACTION];
        t.write(Tree::new(g, probs));
    });
    let mut trees: [Tree; BATCH] = unsafe {
        core::mem::transmute(trees)
    };
    
    let mut leaves = [[0.0f32; STATE]; BATCH];
    trees.par_iter_mut().zip(leaves.par_iter_mut()).for_each(|(tree, s)| {
        todo!()
    });
    
    (0..1000).for_each(|_| {
        let leaves_tensor = dev.tensor(leaves.clone());
        let mut grads = model.alloc_grads();
        grads = {
            let core_tensor = model.forward(leaves_tensor.trace(grads));
            let predicted_logits: Tensor<(usize, usize), f32, Cuda, OwnedTape<f32, Cuda>> = core_tensor.slice((0.., 0..ACTION));
            assert_eq!(predicted_logits.shape(), &(64, 272));

            let p: f32 = (ACTION as f32).recip();
            let target_probs = [[p; ACTION]; BATCH];
            let target_tensor: Tensor<(usize, usize), f32, Cuda> = dev.tensor(target_probs.clone()).slice((0.., 0..));

            let cross_entropy = cross_entropy_with_logits_loss(predicted_logits, target_tensor);
            print!("{:10}\t", cross_entropy.array());
            cross_entropy.backward()
        };
        grads = {
            let core_tensor = model.forward(leaves_tensor.trace(grads));
            let predicted_values: Tensor<(usize, usize), f32, Cuda, OwnedTape<f32, Cuda>> = core_tensor.slice((0.., ACTION..));
            assert_eq!(predicted_values.shape(), &(64, 1));

            let target_values = [[10.0f32; 1]; BATCH];
            let target_values = dev.tensor(target_values.clone()).slice((0.., 0..));

            let square_error = mse_loss(predicted_values, target_values);
            println!("{}", square_error.array());
            square_error.backward()
        };
        opt.update(&mut model, &grads).unwrap();
        model.zero_grads(&mut grads);
    });
    


    // cross_entropy.backward();
    // sgd.update(&mut model_core, &grads).expect("Unused params");

    // let grads = model_core.alloc_grads();
    // let mut sgd: Sgd<Gradients<f32, Cuda>, f32, Cuda> = Sgd::new(
    //     &grads,
    //     SgdConfig {
    //         lr: 1e-1,
    //         momentum: Some(Momentum::Nesterov(0.9)),
    //         weight_decay: None,
    //     },
    // );

    // let leaves_tensor = dev.tensor(leaves.clone());
    // let u = leaves_tensor.ghost();
    // let core_tensor: Tensor<(Const<64>, Const<128>), f32, Cuda, OwnedTape<f32, Cuda>> = model_core.forward(leaves_tensor.trace(grads));
    // let u = core_tensor.ghost();
    // let logits = model_probs.forward(core_tensor);
    // let p: f32 = (ACTION as f32).recip();
    // let target_probs = [[p; ACTION]; BATCH];
    // let target_tensor = dev.tensor(target_probs.clone());
    // let cross_entropy = cross_entropy_with_logits_loss(logits, target_tensor);
    // dbg!(cross_entropy.array());
    // cross_entropy.backward();

    
    // type T = <S as ReduceShape<Axis<1>>>::Output;
    // let probs = logits.softmax::<Axis<1>>().array();
    // dbg!(probs.map(|probs| probs.into_iter().sum::<f32>()));
}
