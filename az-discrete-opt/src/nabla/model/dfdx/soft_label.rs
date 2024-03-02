use dfdx::{shapes::{Const, HasShape, Rank0, Rank1, Rank2}, tensor::{Cuda, CudaError, OnesTensor, Tensor, TensorFrom, ZerosTensor}, tensor_ops::{BroadcastTo, ChooseFrom, SumTo, TryLe, TryMatMul, TryMul, TrySub}};

pub struct SoftLabel<const L: usize> {
    shift: Tensor<Rank2<2, L>, f32, Cuda>,
    scale: Tensor<Rank2<2, L>, f32, Cuda>,
    labels: Tensor<Rank1<L>, f32, Cuda>,
    one: Tensor<Rank0, f32, Cuda>,
    zero: Tensor<Rank0, f32, Cuda>,
}

impl<const L: usize> SoftLabel<L> {
    pub fn try_new(labels: [f32; L], dev: &Cuda) -> Result<Self, CudaError> {
        let mut shift = [[0.0; L]; 2];
        let mut scale = [[0.0; L]; 2];
        for i in 0..(L - 1) {
            shift[0][i] = labels[i + 1];
            shift[1][i + 1] = labels[i];
            scale[0][i] = 1.0 / (labels[i] - labels[i + 1]);
            scale[1][i + 1] = 1.0 / (labels[i + 1] - labels[i]);
        }
        // shift[0][L - 1] = labels[L - 1] + 1.0;
        // scale[0][L - 1] = -1.0;
        // shift[1][0] = labels[0] - 1.0;
        // scale[1][0] = 1.0;
        Ok(Self {
            shift: dev.try_tensor(shift)?,
            scale: dev.try_tensor(scale)?,
            labels: dev.try_tensor(labels)?,
            one: dev.try_ones::<Rank0>()?, //.try_broadcast()?,
            zero: dev.try_zeros::<Rank0>()?, //.try_broadcast()?,
        })
    }

    pub fn new(labels: [f32; L], dev: &Cuda) -> Self {
        Self::try_new(labels, dev).unwrap()
    }
}

impl<const L: usize> SoftLabel<L> {
    pub fn try_label(&self, input: Tensor<(usize,), f32, Cuda>) -> Result<Tensor<(usize, Const<L>), f32, Cuda>, CudaError> {
        let Self { shift, scale, labels: _, one, zero } = self;
        let d = input.shape().0;
        // calculate soft label weights
        let shift = shift.clone().try_broadcast_like(&(Const, d, Const))?;
        let scale = scale.clone().try_broadcast_like(&(Const, d, Const))?;
        let input = input.clone().try_broadcast_like(&(Const, d, Const))?;
        let input = input.try_sub(shift)?;
        let input = input.try_mul(scale)?;
        // remove negative values
        // let input: Tensor<Rank3<2, D, L>, f32, Cuda> = input.try_relu()?;
        // // remove values greater than 1
        let ones = one.clone().try_broadcast_like(&(Const, d, Const))?;
        let cmp = input.try_le(&ones)?;
        let zeros = zero.clone().try_broadcast_like(&(Const, d, Const))?;
        let input = cmp.try_choose(input, zeros)?;
        // sum the weights
        let input = input.try_sum()?;
        let input = input.try_clamp(0., 1.)?;
        Ok(input)
    }

    pub fn label(&self, input: Tensor<(usize,), f32, Cuda>) -> Tensor<(usize, Const<L>), f32, Cuda> {
        self.try_label(input).unwrap()
    }

    pub fn try_unlabel(&self, input: Tensor<(usize, Const<L>), f32, Cuda>) -> Result<Tensor<(usize,), f32, Cuda>, CudaError> {
        let Self { shift: _, scale: _, labels, one: _, zero: _ } = self;
        input.try_matmul(labels.clone())
        
    }

    pub fn unlabel(&self, input: Tensor<(usize, Const<L>), f32, Cuda>) -> Tensor<(usize,), f32, Cuda> {
        self.try_unlabel(input).unwrap()
    }
}

// #[test]
// fn softly_labels_some_f32_numbers() {
//     const L: usize = 5;
//     let labels: [f32; L] = [-1., 0., 0.5, 1., 2.];
//     let dev: Cuda = AutoDevice::default();
//     let labels = SoftLabel::new(labels, &dev);
//     const N: usize = 11;
//     let nums: [f32; N] = [
//         -2.0,
//         -1.0,
//         -0.5,
//         0.,
//         0.25,
//         0.5,
//         0.75,
//         1.,
//         1.5,
//         2.0,
//         3.0,
//     ];
//     let nums = dev.tensor(nums);
//     let nums = nums.reshape_like(&(N,));
//     let labeled_nums = labels.label(nums);
//     let labeled_nums = labeled_nums.reshape_like(&(Const, Const));
//     let labeled_nums: [[f32; L]; N] = labeled_nums.array();
//     dbg!(&labeled_nums);
//     let correct_labels = [
//         [0.0, 0.0, 0.0, 0.0, 0.0], // -2.0
//         [1.0, 0.0, 0.0, 0.0, 0.0], // -1.0
//         [0.5, 0.5, 0.0, 0.0, 0.0], // -0.5
//         [0.0, 1.0, 0.0, 0.0, 0.0], //  0.0
//         [0.0, 0.5, 0.5, 0.0, 0.0], //  0.25
//         [0.0, 0.0, 1.0, 0.0, 0.0], //  0.5
//         [0.0, 0.0, 0.5, 0.5, 0.0], //  0.75
//         [0.0, 0.0, 0.0, 1.0, 0.0], //  1.0
//         [0.0, 0.0, 0.0, 0.5, 0.5], //  1.5
//         [0.0, 0.0, 0.0, 0.0, 1.0], //  2.0
//         [0.0, 0.0, 0.0, 0.0, 0.0], //  3.0
//     ];
//     for (i, (labels, correct_labels)) in labeled_nums.iter().zip(correct_labels.iter()).enumerate() {
//         debug_assert_eq!(labels, correct_labels, "i = {i}");
//     }
// }
