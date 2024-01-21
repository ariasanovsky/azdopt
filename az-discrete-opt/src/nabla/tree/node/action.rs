// use core::num::NonZeroUsize;

// pub struct ActionData {
//     pub(crate) a: usize,
//     pub(crate) next_position: Option<NextPositionData>,
//     pub(crate) h_theta_sa: f32,
// }

// pub(crate) struct NextPositionData {
//     pub(crate) next_position: NonZeroUsize,
//     pub(crate) h_t_sa: f32,
// }

// impl ActionData {
//     pub(crate) fn new_predicted(a: usize, h_sa: f32) -> Self {
//         Self {
//             a,
//             next_position: None,
//             h_theta_sa: h_sa,
//         }
//     }

//     // pub(crate) fn action(&self) -> usize {
//     //     // debug_assert!(self.g_sa().is_some());
//     //     self.a
//     // }

//     // pub(crate) fn next_position_mut(&mut self) -> &mut Option<NonZeroU32> {
//     //     todo!()
//     //     // debug_assert!(self.g_sa().is_some());
//     //     // &mut self.next_position
//     // }

//     // pub(crate) fn h_theta_sa(&self) -> f32 {
//     //     self.h_theta_sa
//     // }

//     // pub(crate) fn update_h_sa(&mut self, h: f32) {
//     //     self.h_sa = self.h_sa.max(h)
//     // }
// }
