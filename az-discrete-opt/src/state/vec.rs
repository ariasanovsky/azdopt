// use super::prohibit::WithProhibitions;

// pub trait StateVec<A> {
//     const DIM: usize;
//     fn write_vec(&self, vec: &mut [f32]);
// }

// impl<A: Action<S>, S: StateVec<A>> StateVec<A> for WithProhibitions<S> {
//     const DIM: usize = S::DIM + A::DIM;

//     fn write_vec(&self, vec: &mut [f32]) {
//         let (state_vec, action_vec) = vec.split_at_mut(S::DIM);
//         debug_assert!(action_vec.len() == A::DIM);
//         self.state.write_vec(state_vec);
//         action_vec.fill(0.0);
//         self.prohibited_actions.iter().for_each(|i| action_vec[*i] = 1.0);
//         // action_vec.iter_mut().enumerate().for_each(|(i, f)| {
//         //     *f = match self.prohibited_actions.contains(&i) {
//         //         true => 1.,
//         //         false => 0.,
//         //     }
//         // });
//     }
// }
