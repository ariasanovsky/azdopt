use dfdx::{
    losses::huber_loss,
    optim::{Momentum, Sgd, SgdConfig},
    prelude::*,
    tensor::AutoDevice,
};

/* (S, A) is as follows:
    * each s in S is equivalent to the following triple (g, p, t) where:
        * g is a 2-edge-colored complete graph on N = 17 vertices
        * a is the set of recoloring actions that may be taken from s
        * t is the number of remaining recolorings before termination
    * each a in A is a pair (e, c) where
        * e is an edge of g
        * c is a color (red or blue)
    * taking action a = (e, c) from state s = (g, p, t) results in the state s' = (g', p', t') where:
        * g' is the graph obtained from g by recoloring e with c
        * t' = t - 1
        * all actions (e, c') are removed from p'; furthermore if t' = 0, then p' is emptied
            * i.e., the same edge may not be recolored, and no actions may be taken if t' = 0
    * to evaluate s, we convert it into a tensor product of the following vectors:
        * c_red, c_blue where the uv-th entry equals:
            * the number of K_4's in g_red (g_blue) + uv
        * e_red, e_blue where the uv-th entry equals:
            * 1.0 or 0.0 if the uv-th edge is red (blue)
        * p_red, p_blue where the uv-th entry equals:
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
const HIDDEN_2: usize = 64;
const PREDICTION: usize = ACTION + 1;

type Architecture = (
    (Linear<STATE, HIDDEN_1>, ReLU),
    (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
    Linear<HIDDEN_2, PREDICTION>,
);

// search constants
const BATCH: usize = 64;




fn main() {
    let dev = AutoDevice::default();
    let mut model = dev.build_module::<Architecture, f32>();
    let mut grads = model.alloc_grads();
    let mut sgd: Sgd<Gradients<f32, Cuda>, f32, Cuda> = Sgd::new(
        &grads,
        SgdConfig {
            lr: 1e-1,
            momentum: Some(Momentum::Nesterov(0.9)),
            weight_decay: None,
        },
    );
}
