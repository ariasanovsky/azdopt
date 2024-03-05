use dfdx::{shapes::{Axes, ConstShape, Rank0, ReduceShape, ReduceShapeTo, Shape}, tensor::{Tape, Tensor}, tensor_ops::{BroadcastTo, ChooseFrom, Device, MeanTo}};

// pub fn adjust_logits_with_choice_and_sqrt_supp_old<const B: usize, const A: usize, D: Device<f32>, T: Tape<f32, D>>(
//     predicted_logits: Tensor<Rank2<B, A>, f32, D, T>,
//     choice: Tensor<Rank2<B, A>, bool, D>,
//     num_chosen: Tensor<Rank1<B>, f32, D>,
//     mask_value: Tensor<Rank0, f32, D>,
// ) -> Tensor<Rank2<B, A>, f32, D, T> {
//     let masked_logits = choice.choose(predicted_logits, mask_value.broadcast());
//     let scaled_logits = masked_logits * num_chosen.sqrt().recip().broadcast();
//     scaled_logits
// }

pub fn adjust_logits_with_choice_and_sqrt_supp<S: ConstShape, D: Device<f32>, T: Tape<f32, D>, Dst: Shape, Ax: Axes>(
    predicted_logits: Tensor<S, f32, D, T>,
    choice: Tensor<S, bool, D>,
    num_chosen: Tensor<Dst, f32, D>,
    mask_value: Tensor<Rank0, f32, D>,
) -> Tensor<S, f32, D, T>
where
    S: ReduceShapeTo<Dst, Ax>,
{
    let masked_logits = choice.choose(predicted_logits, mask_value.broadcast());
    let scaled_logits = masked_logits * num_chosen.sqrt().recip().broadcast();
    scaled_logits
}

pub fn masked_logit_cross_entropy<S: ConstShape, D: Device<f32>, T: Tape<f32, D>, Dst: ConstShape, Ax: Axes>(
    predicted_logits: Tensor<S, f32, D, T>,
    target_probs: Tensor<S, f32, D>,
    choice: Tensor<S, bool, D>,
    num_chosen: Tensor<Dst, f32, D>,
    mask_value: Tensor<Rank0, f32, D>,
) -> Tensor<Dst, f32, D, T>
where
    S: ReduceShape<Ax> + ReduceShapeTo<Dst, Ax>,
{
    let adjusted_logits = adjust_logits_with_choice_and_sqrt_supp(predicted_logits, choice, num_chosen, mask_value);
    let predicted_log_probs = adjusted_logits.log_softmax::<Ax>();
    let mean_rescale = (Dst::NUMEL as f64).recip() as f32;
    let losses: Tensor<Dst, f32, D, T> = (predicted_log_probs * target_probs).mean().negate() / mean_rescale;
    losses
}

// pub fn masked_logit_cross_entropy_old<const B: usize, const A: usize, D: Device<f32>, T: Tape<f32, D>>(
//     predicted_logits: Tensor<Rank2<B, A>, f32, D, T>,
//     target_probs: Tensor<Rank2<B, A>, f32, D>,
//     choice: Tensor<Rank2<B, A>, bool, D>,
//     num_chosen: Tensor<Rank1<B>, f32, D>,
//     mask_value: Tensor<Rank0, f32, D>,
// ) -> Tensor<Rank1<B>, f32, D, T> {
//     let adjusted_logits = adjust_logits_with_choice_and_sqrt_supp(predicted_logits, choice, num_chosen, mask_value);
//     // let masked_logits = choice.choose(predicted_logits, mask_value.broadcast());
//     // let scaled_logits = masked_logits * num_chosen.sqrt().recip().broadcast();
//     let predicted_log_probs = adjusted_logits.log_softmax::<<Rank2<B, A> as Shape>::LastAxis>();
//     let mean_rescale = (B as f64).recip() as f32;
//     let losses: Tensor<Rank1<B>, f32, D, T> = (predicted_log_probs * target_probs).mean().negate() / mean_rescale;
//     losses
// }

#[cfg(test)]
mod tests {
    use dfdx::{nn::{builders::Linear, modules::ReLU, DeviceBuildExt, Module, ZeroGrads}, optim::{Adam, AdamConfig, Optimizer}, shapes::Rank2, tensor::{AsArray, AutoDevice, OnesTensor, TensorFrom, Trace}, tensor_ops::{Backward, SumTo}};

    use super::*;
    
    #[test]
    fn masked_cross_entropy_loss_converges_on_support() {
        const B: usize = 3;
        const STATE: usize = 5;
        const ACTION: usize = 7;

        let target_probs: [[f32; ACTION]; B] = [
            [0.125, 0., 0.5, 0.125, 0.25, 0., 0.],
            [0., 0.125, 0., 0., 0.5, 0.125, 0.25],
            [0., 0.375, 0.625, 0., 0., 0., 0.]
        ];

        let choice = [
            [true, false, true, true, true, false, false],
            [false, true, false, false, true, true, true],
            [false, true, true, false, false, false, false]
        ];
        let num_chosen = [4., 4., 2.];

        let dev = AutoDevice::default();
        const HIDDEN_1: usize = 10;
        const HIDDEN_2: usize = 10;
        type M = (
            (Linear<STATE, HIDDEN_1>, ReLU),
            (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
            Linear<HIDDEN_2, ACTION>,
        );
        let mut m = dev.build_module::<M, f32>();
        let mut grads = m.alloc_grads();
        let input = [
            [0.1, 0.2, 0.3, 0.4, 0.5],
            [0.5, 0.4, 0.3, 0.2, 0.1],
            [0.1, 0.2, 0.3, 0.4, 0.5],
        ];
        let mask_value = dev.ones() * f32::MIN;
        let choice_dev = dev.tensor(choice);
        let num_chosen = dev.tensor(num_chosen);

        let cfg = AdamConfig {
            lr: 5e-3,
            betas: [0.9, 0.999],
            eps: 1e-8,
            weight_decay: Some(dfdx::optim::WeightDecay::L2(1e-6)),
        };
        let mut opt: Adam<_, f32, _> = Adam::new(&m, cfg);

        for i in 0..150 {
            let input = dev.tensor(input).trace(grads);
            let predicted_logits = m.forward(input);
            let target_probs = dev.tensor(target_probs);
            let loss = masked_logit_cross_entropy(predicted_logits, target_probs, choice_dev.clone(), num_chosen.clone(), mask_value.clone());
            dbg!(i, loss.array());
            grads = loss.sum().backward();
            opt.update(&mut m, &mut grads).unwrap();
            m.zero_grads(&mut grads);
        }
        let input = dev.tensor(input);
        let predicted_logits = m.forward(input);
        let masked_logits = choice_dev.choose(predicted_logits, mask_value.broadcast());
        let rescaled_logits = masked_logits * num_chosen.sqrt().recip().broadcast();
        let predicted_probs = rescaled_logits.softmax::<<Rank2<B, ACTION> as Shape>::LastAxis>();
        let predicted_probs = predicted_probs.array();
        for i in 0..B {
            for j in 0..ACTION {
                if choice[i][j] {
                    assert!(
                        (predicted_probs[i][j] - target_probs[i][j]).abs() < 1e-2,
                        "i: {}, j: {}, target: {}, predicted: {}",
                        i,
                        j,
                        target_probs[i][j],
                        predicted_probs[i][j]
                    );
                }
            }
        }
    }
}