pub trait Cost<F = f32> {
    fn evaluate(&self) -> F;
}

// pub trait CostsOneEach {}

// impl<T: CostsOneEach> Cost for Vec<T> {
//     fn cost(&self) -> f32 {
//         self.len() as f32
//     }
// }

// // impl Cost<f32> and Cost<f64> for all numeric types
// macro_rules! impl_cost {
//     ($($t:ty),*) => {
//         $(
//             impl Cost for $t {
//                 fn cost(&self) -> f32 {
//                     *self as f32
//                 }
//             }
//         )*
//     };
// }

// impl_cost!(u8, u16, u32, u64, u128, usize, i8, i16, i32, i64, i128, isize, f32, f64);

// impl<F: core::ops::Add<Output = F>, C: Cost<F>, D: Cost<F>> Cost<F> for (C, D) {
//     fn cost(&self) -> F {
//         self.0.cost() + self.1.cost()
//     }
// }
