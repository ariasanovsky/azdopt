

use dfdx::{
    // losses::cross_entropy_with_logits_loss,
    losses::cross_entropy_with_logits_loss, nn::{BuildOnDevice, DeviceBuildExt, ZeroGrads}, prelude::{Gradients, Rank2}, shapes::{Const, Rank0, Rank1, Rank3, Shape}, tensor::{Cuda, OnesTensor, Tensor, ZerosTensor}, tensor_ops::{ReshapeTo, SumTo}
};

use dfdx::{
    nn::Module,
    optim::{Adam, Optimizer},
    tensor::{AsArray, OwnedTape, Trace},
    tensor_ops::{AdamConfig, Backward},
};

use crate::nabla::model::dfdx::masked_logits::{adjust_logits_with_choice_and_sqrt_supp, masked_logit_cross_entropy};

use self::soft_label::SoftLabel;

use super::NablaModel;

pub mod masked_logits;
pub mod soft_label;

// pub struct HardActionModel<M, const BATCH: usize, const STATE: usize, const ACTION: usize>
// where
//     M: BuildOnDevice<Cuda, f32>,
// {
//     model: <M as BuildOnDevice<Cuda, f32>>::Built,
//     gradients: Option<Gradients<f32, Cuda>>,
//     optimizer: Adam<<M as BuildOnDevice<Cuda, f32>>::Built, f32, Cuda>,
//     states_dev: Tensor<Rank2<BATCH, STATE>, f32, Cuda>,
//     action_weights_dev: Tensor<Rank2<BATCH, ACTION>, f32, Cuda>,
//     // state_weights_dev: Tensor<Rank1<BATCH>, f32, Cuda>,
//     actions_dev: Tensor<Rank2<BATCH, ACTION>, f32, Cuda>,
// }

// impl<M, const BATCH: usize, const STATE: usize, const ACTION: usize>
//     HardActionModel<M, BATCH, STATE, ACTION>
// where
//     M: BuildOnDevice<Cuda, f32>,
// {
//     pub fn new(dev: Cuda, cfg: AdamConfig) -> Self {
//         let model = dev.build_module::<M, _>();
//         let gradients = Some(model.alloc_grads());
//         let states_dev = dev.zeros();
//         let action_weights_dev = dev.zeros();
//         // let state_weights_dev = dev.zeros();
//         let actions_dev = dev.zeros();
//         let optimizer = Adam::new(&model, cfg);
//         Self {
//             model,
//             gradients,
//             optimizer,
//             states_dev,
//             action_weights_dev,
//             // state_weights_dev,
//             actions_dev,
//         }
//     }
// }

// impl<M, const BATCH: usize, const STATE: usize, const ACTION: usize> NablaModel
//     for HardActionModel<M, BATCH, STATE, ACTION>
// where
//     M: BuildOnDevice<Cuda, f32>,
//     <M as dfdx::nn::BuildOnDevice<dfdx::tensor::Cuda, f32>>::Built: Module<
//         Tensor<Rank2<BATCH, STATE>, f32, Cuda>,
//         Output = Tensor<Rank2<BATCH, ACTION>, f32, Cuda>,
//     >,
//     <M as dfdx::nn::BuildOnDevice<dfdx::tensor::Cuda, f32>>::Built: Module<
//         Tensor<Rank2<BATCH, STATE>, f32, Cuda, OwnedTape<f32, Cuda>>,
//         Output = Tensor<Rank2<BATCH, ACTION>, f32, Cuda, OwnedTape<f32, Cuda>>,
//     >,
// {
//     fn write_predictions(&mut self, states: &[f32], predictions: &mut [f32]) {
//         let Self {
//             model,
//             gradients: _,
//             optimizer: _,
//             states_dev,
//             action_weights_dev: _,
//             actions_dev,
//         } = self;
//         debug_assert_eq!(states.len(), BATCH * STATE);
//         debug_assert_eq!(predictions.len(), BATCH * ACTION);
//         states_dev.copy_from(states);
//         *actions_dev = model.forward(states_dev.clone());
//         actions_dev.copy_into(predictions);
//     }

//     fn update_model(
//         &mut self,
//         states: &[f32],
//         observations: &[f32],
//     ) -> f32 {
//         let Self {
//             model,
//             gradients,
//             optimizer,
//             states_dev,
//             action_weights_dev,
//             actions_dev,
//         } = self;
//         debug_assert_eq!(states.len(), BATCH * STATE);
//         debug_assert_eq!(observations.len(), BATCH * ACTION);
//         states_dev.copy_from(states);
//         actions_dev.copy_from(observations);
//         let weight_sum: f32 = observations.iter().sum::<f32>();
//         dbg!(weight_sum);
//         // let action_normalization = action_weights_dev.clone().sum::<Rank1<BATCH>, _>().recip();
//         // let action_normalization: Tensor<Rank2<BATCH, ACTION>, _, _> = action_normalization.broadcast();
//         *action_weights_dev = action_weights_dev.clone() / weight_sum;
//         // let state_normalization = state_weights_dev.clone().sum::<Rank0, _>().recip();
//         // let state_normalization: Tensor<Rank1<BATCH>, _, _> = state_normalization.broadcast();
//         // *state_weights_dev = state_weights_dev.clone() * state_normalization;
//         let gradients = gradients.take().unwrap_or_else(|| model.alloc_grads());
//         let states_traced = states_dev.clone().trace(gradients);
//         let prediction_logits = model.forward(states_traced);

//         // let loss = cross_entropy_with_logits_loss(prediction_logits, actions_dev.clone());
//         let error = (prediction_logits - actions_dev.clone()).square();
//         let loss = (
//             (error * action_weights_dev.clone()).sum::<Rank1<BATCH>, _>()
//             // * state_weights_dev.clone()
//         )
//         .sum::<Rank0, _>();
//         let l: f32 = loss.array();
//         dbg!(l);
//         let mut grads = loss.backward();
//         optimizer.update(model, &mut grads).unwrap();
//         model.zero_grads(&mut grads);
//         self.gradients = Some(grads);
//         l
//     }
// }

pub struct SoftActionModel<M, const B: usize, const S: usize, const A: usize, const L: usize, const AL: usize>
where
    M: BuildOnDevice<Cuda, f32>,
{
    model: <M as BuildOnDevice<Cuda, f32>>::Built,
    gradients: Option<Gradients<f32, Cuda>>,
    optimizer: Adam<<M as BuildOnDevice<Cuda, f32>>::Built, f32, Cuda>,
    batch_state: Tensor<Rank2<B, S>, f32, Cuda>,
    // batch_action_label: Tensor<Rank3<BATCH, ACTION, LABEL>, f32, Cuda>,
    batch_action: Tensor<Rank2<B, A>, f32, Cuda>,
    // label_one: Tensor<Rank2<L, 1>, f32, Cuda>,
    label: SoftLabel<L>,
    logit_mask_value: Tensor<Rank0, f32, Cuda>,
    valid_actions_dev: Tensor<Rank2<B, A>, bool, Cuda>,
    num_actions_dev: Tensor<Rank1<B>, f32, Cuda>,
}

impl<M, const B: usize, const S: usize, const A: usize, const L: usize, const AL: usize>
    SoftActionModel<M, B, S, A, L, AL>
where
    M: BuildOnDevice<Cuda, f32>,
{
    pub fn new(dev: Cuda, cfg: AdamConfig, labels: [f32; L]) -> Self {
        let model = dev.build_module::<M, _>();
        let gradients = Some(model.alloc_grads());
        let batch_state = dev.zeros();
        // let batch_action_label = dev.zeros();
        let batch_action = dev.zeros();
        // let label_one = dev.ones();
        let label = SoftLabel::new(labels, &dev);
        let optimizer = Adam::new(&model, cfg);
        Self {
            model,
            gradients,
            optimizer,
            batch_state,
            // batch_action_label,
            batch_action,
            // label_one,
            label,
            logit_mask_value: dev.ones() * f32::MIN,
            valid_actions_dev: dev.zeros(),
            num_actions_dev: dev.zeros(),
        }
    }
}

impl<M, const B: usize, const S: usize, const A: usize, const L: usize, const AL: usize> NablaModel
    for SoftActionModel<M, B, S, A, L, AL>
where
    M: BuildOnDevice<Cuda, f32>,
    <M as dfdx::nn::BuildOnDevice<dfdx::tensor::Cuda, f32>>::Built: Module<
        Tensor<Rank2<B, S>, f32, Cuda>,
        Output = (Tensor<Rank2<B, AL>, f32, Cuda>, Tensor<Rank2<B, A>, f32, Cuda>),
    >,
    <M as dfdx::nn::BuildOnDevice<dfdx::tensor::Cuda, f32>>::Built: Module<
        Tensor<Rank2<B, S>, f32, Cuda, OwnedTape<f32, Cuda>>,
        Output = (Tensor<Rank2<B, AL>, f32, Cuda, OwnedTape<f32, Cuda>>, Tensor<Rank2<B, A>, f32, Cuda, OwnedTape<f32, Cuda>>),
    >,
{
    fn write_predictions(
        &mut self,
        states: &[f32],
        valid_actions: &[bool],
        num_actions: &[f32],
        v_predictions: &mut [f32],
        p_predictions: &mut [f32],
    ) {
        let Self {
            model,
            gradients: _,
            optimizer: _,
            batch_state,
            // batch_action_label,
            batch_action,
            // label_one,
            label,
            logit_mask_value,
            valid_actions_dev,
            num_actions_dev,
        } = self;
        debug_assert_eq!(states.len(), B * S);
        debug_assert_eq!(valid_actions.len(), B * A);
        debug_assert_eq!(num_actions.len(), B);
        debug_assert_eq!(v_predictions.len(), B * A);
        debug_assert_eq!(p_predictions.len(), B * A);
        batch_state.copy_from(states);
        valid_actions_dev.copy_from(valid_actions);
        num_actions_dev.copy_from(num_actions);
        let factor = (L as f64).sqrt().recip() as f32;
        // let predictions = match model.forward(batch_state.clone()) {
        //     m => m,
        //     // _ => unreachable!(),
        // };
        let predictions = model.forward(batch_state.clone());
        let v_label_predictions = predictions.0;
        let predicted_prob_logits = predictions.1;
        let predicted_value_label_logits = v_label_predictions.reshape::<Rank3<B, A, L>>() * factor;
        let predicted_value_label_probs =
            predicted_value_label_logits
            .softmax::<<Rank3<B, A, L> as Shape>::LastAxis>();
        // let v_predictions_dev: Tensor<Rank2<B, A>, f32, Cuda> = (predicted_value_label_probs).matmul(label_one.clone()).reshape();
        let d = B * A;
        let predicted_value_label_probs = predicted_value_label_probs.reshape_like(&(d, Const));
        let v_predictions_dev = label.unlabel(predicted_value_label_probs);
        let v_predictions_dev: Tensor<Rank2<B, A>, f32, Cuda> = v_predictions_dev.reshape_like(&(Const, Const));
        v_predictions_dev.copy_into(v_predictions);
        let predicted_prob_logits = adjust_logits_with_choice_and_sqrt_supp(
            predicted_prob_logits,
            valid_actions_dev.clone(),
            num_actions_dev.clone(),
            logit_mask_value.clone(),
        );
        let p_predictions_dev = predicted_prob_logits.softmax::<<Rank2<B, A> as Shape>::LastAxis>();
        p_predictions_dev.copy_into(p_predictions);
    }

    fn update_model(
        &mut self,
        states: &[f32],
        valid_actions: &[bool],
        num_actions: &[f32],
        v_observations: &[f32],
        n_observations: &[f32],
    )
        -> f32 {
        let Self {
            model,
            gradients,
            optimizer,
            batch_state,
            // batch_action_label,
            batch_action,
            // label_one: _,
            label,
            logit_mask_value,
            valid_actions_dev,
            num_actions_dev,
        } = self;
        debug_assert_eq!(states.len(), B * S);
        debug_assert_eq!(valid_actions.len(), B * A);
        debug_assert_eq!(num_actions.len(), B);
        debug_assert_eq!(v_observations.len(), B * A);
        debug_assert_eq!(n_observations.len(), B * A);
        batch_state.copy_from(states);
        valid_actions_dev.copy_from(valid_actions);
        num_actions_dev.copy_from(num_actions);
        batch_action.copy_from(v_observations);
        let d = B * A;
        let batch_action = batch_action.clone().reshape_like(&(d,));
        let observed_label_probs = label.label(batch_action);
        let observed_label_probs: Tensor<Rank3<B, A, L>, f32, Cuda> = observed_label_probs.reshape_like(&(Const, Const, Const));
        let num_observations = observed_label_probs.clone().sum::<Rank1<B>, _>();
        use dfdx::tensor_ops::{MinTo, MaxTo, MeanTo, BroadcastTo};
        let min_num_obs = num_observations.clone().min::<Rank0, _>().array();
        let max_num_obs = num_observations.clone().max::<Rank0, _>().array();
        let mean_num_obs = num_observations.clone().mean::<Rank0, _>().array();
        dbg!(min_num_obs, max_num_obs, mean_num_obs);
        let gradients = gradients.take().unwrap_or_else(|| model.alloc_grads());
        let batch_state_traced = batch_state.clone().trace(gradients);
        let (predicted_value_label_logits, predicted_prob_logits) = model.forward(batch_state_traced);
        let factor = (L as f64).sqrt().recip() as f32;
        let predicted_value_label_logits = predicted_value_label_logits.reshape::<Rank3<B, A, L>>() * factor;
        // TODO! weight the loss for (s, a) with something like sqrt(n_observations[s, a])
        let value_loss = cross_entropy_with_logits_loss(predicted_value_label_logits, observed_label_probs);
        dbg!(value_loss.array());
        let batch_action = &mut self.batch_action;
        batch_action.copy_from(n_observations);
        let num_observations = batch_action.clone().sum::<Rank1<B>, _>().recip();
        let probs_dev = batch_action.clone() * num_observations.broadcast();
        // TODO! mask
        // let probs_loss = cross_entropy_with_logits_loss(predicted_prob_logits, probs_dev);
        
        let probs_loss = masked_logit_cross_entropy(
            predicted_prob_logits,
            probs_dev,
            valid_actions_dev.clone(),
            num_actions_dev.clone(),
            logit_mask_value.clone(),
        );

        dbg!(probs_loss.array());
        let total_loss = value_loss + probs_loss.sum();
        let loss = total_loss.array();
        let mut grads = total_loss.backward();
        optimizer.update(model, &mut grads).unwrap();
        model.zero_grads(&mut grads);
        self.gradients = Some(grads);
        dbg!(loss)
    }
}

#[test]
fn split_into_2() {
    use dfdx::{prelude::{AutoDevice, mse_loss}, tensor::TensorFrom};
    let dev = AutoDevice::default();
    type H1 = dfdx::nn::builders::Linear<5, 3>;
    type H2 = dfdx::nn::builders::Linear<5, 2>;
    type M = dfdx::nn::modules::SplitInto<(H1, H2)>;
    let mut m = dev.build_module::<M, f32>();
    let mut grads = m.alloc_grads();
    let input = [0., 1., 2., 3., 4.];
    let desired_outputs = ([-10., 13., 11.], [10., -13.]);

    let cfg = AdamConfig {
        lr: 5e-3,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: Some(dfdx::optim::WeightDecay::L2(1e-6)),
    };

    for _ in 0..500 {
        let input = dev.tensor(input.clone());
        let (desired_output_1, desired_output_2) = (dev.tensor(desired_outputs.0), dev.tensor(desired_outputs.1));
        let traced_input = input.trace(grads);
        let (output_1, output_2) = m.forward(traced_input);
        let loss_1 = mse_loss(output_1, desired_output_1);
        let loss_2 = mse_loss(output_2, desired_output_2);
        let loss = loss_1 + loss_2;
        dbg!(loss.array());
        grads = loss.backward();
        let mut opt: Adam<_, f32, _> = Adam::new(&m, cfg);
        opt.update(&mut m, &mut grads).unwrap();
        m.zero_grads(&mut grads);
    }

    let input = dev.tensor(input);
    let (output_1, output_2) = m.forward(input);
    dbg!(output_1.array(), output_2.array());

    let input_batch = [
        [0., 1., 2., 3., 4.],
        [5., 6., 7., 8., 9.],
    ];
    let input = dev.tensor(input_batch);
    let (output_1, output_2) = m.forward(input);
    let (output_1, output_2) = (output_1.array(), output_2.array());
    dbg!(output_1, output_2);

    let foo = [0.0f32, 1., 0., -0., 1.];
    let bar = f32::INFINITY;
    let foo = dev.tensor(foo);
    let foo = foo * bar;
    let foo = foo.array();
    assert_eq!(foo, [0.0f32, f32::INFINITY, 0., -0., f32::INFINITY]);
}

#[test]
fn entropy_loss_with_softmax_mask() {
    use dfdx::{prelude::{AutoDevice, cross_entropy_with_logits_loss}, tensor::TensorFrom};
    let dev = AutoDevice::default();
    let mask_value = f32::MIN;
    let mask = [0., mask_value, 0., mask_value, 0., mask_value];
    let mask = dev.tensor(mask);
    let target = [0.125f32, 0., 0.875, 0., 0., 0.];
    let target = dev.tensor(target);

    const STATE: usize = 100;
    const HIDDEN_1: usize = 1024;
    const HIDDEN_2: usize = 1024;
    const OUTPUT: usize = 6;
    type M = (
        (dfdx::nn::builders::Linear<STATE, HIDDEN_1>, dfdx::nn::builders::ReLU),
        (dfdx::nn::builders::Linear<HIDDEN_1, HIDDEN_2>, dfdx::nn::builders::ReLU),
        dfdx::nn::builders::Linear<HIDDEN_2, OUTPUT>,
    );
    let mut m = dev.build_module::<M, f32>();
    let mut grads = m.alloc_grads();
    let input: [f32; STATE] = core::array::from_fn(|i| i as f32);

    let cfg = AdamConfig {
        lr: 5e-3,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: Some(dfdx::optim::WeightDecay::L2(1e-6)),
    };
    let mut opt: Adam<_, f32, _> = Adam::new(&m, cfg);

    for i in 0..150 {
        let input = dev.tensor(input);
        let input = input.trace(grads);
        let output = m.forward(input);
        let output = output + mask.clone();
        let output = output / 3.0;
        let loss = cross_entropy_with_logits_loss(output, target.clone());
        dbg!(i, loss.array());
        grads = loss.backward();
        opt.update(&mut m, &mut grads).unwrap();
        m.zero_grads(&mut grads);
    }    
}

#[test]
fn entropy_loss_with_choose() {
    use dfdx::{prelude::{AutoDevice, cross_entropy_with_logits_loss}, tensor::TensorFrom, tensor_ops::{BroadcastTo, ChooseFrom}};
    let dev = AutoDevice::default();
    let choice_mask = [true, false, true, false, true, false];
    let choice_mask = dev.tensor(choice_mask);
    let target = [0.125f32, 0., 0.875, 0., 0., 0.];
    let target = dev.tensor(target);

    const STATE: usize = 100;
    const HIDDEN_1: usize = 1024;
    const HIDDEN_2: usize = 1024;
    const OUTPUT: usize = 6;
    type M = (
        (dfdx::nn::builders::Linear<STATE, HIDDEN_1>, dfdx::nn::builders::ReLU),
        (dfdx::nn::builders::Linear<HIDDEN_1, HIDDEN_2>, dfdx::nn::builders::ReLU),
        dfdx::nn::builders::Linear<HIDDEN_2, OUTPUT>,
    );
    let mut m = dev.build_module::<M, f32>();
    let mut grads = m.alloc_grads();
    let input: [f32; STATE] = core::array::from_fn(|i| i as f32);

    let cfg = AdamConfig {
        lr: 5e-3,
        betas: [0.9, 0.999],
        eps: 1e-8,
        weight_decay: Some(dfdx::optim::WeightDecay::L2(1e-6)),
    };
    let mut opt: Adam<_, f32, _> = Adam::new(&m, cfg);

    let mask: Tensor<Rank0, _, Cuda> = dev.ones() * f32::MIN;

    for i in 0..150 {
        let input = dev.tensor(input);
        let input = input.trace(grads);
        let output = m.forward(input);
        let output = choice_mask.clone().choose(output, mask.clone().broadcast());
        let output = output / 3.0;
        let loss = cross_entropy_with_logits_loss(output, target.clone());
        dbg!(i, loss.array());
        grads = loss.backward();
        opt.update(&mut m, &mut grads).unwrap();
        m.zero_grads(&mut grads);
    }    
}
