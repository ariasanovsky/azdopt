// use core::num::NonZeroUsize;

// use crate::nabla::space::NablaStateActionSpace;

// use self::action::ActionData;

pub mod action;

// pub struct StateNode {
//     pub(crate) c: f32,
//     pub(crate) c_star: f32,
//     pub(crate) actions: Box<[ActionData]>,
//     pub(crate) n_s: NumVisits,
// }

// pub(crate) enum NumVisits {
//     Exhausted,
//     Count(NonZeroUsize),
// }

// impl Default for NumVisits {
//     fn default() -> Self {
//         Self::Count(unsafe { NonZeroUsize::new_unchecked(1) })
//     }
// }

// impl StateNode {
//     pub(crate) fn new<Space: NablaStateActionSpace>(
//         space: &Space,
//         state: &Space::State,
//         cost: &Space::Cost,
//         h_theta: &[f32],
//     ) -> Self {
//         let c = space.evaluate(cost);
//         Self {
//             c,
//             c_star: c,
//             actions: space
//                 .action_data(state)
//                 .map(|(a, r)| {
//                     let g_sa = space.g_theta_star_sa(c, r, h_theta[a]);
//                     ActionData::new_predicted(a, g_sa)
//                 })
//                 .collect(),
//             n_s: NumVisits::default(),
//         }
//     }

//     pub(crate) fn is_exhausted(&self) -> bool {
//         matches!(self.n_s, NumVisits::Exhausted)
//     }

//     pub(crate) fn new_exhausted(c: f32) -> Self {
//         Self {
//             c,
//             c_star: c,
//             actions: Default::default(),
//             n_s: NumVisits::Exhausted,
//         }
//     }

//     pub(crate) fn update_h_t_sa(&mut self, action_position: usize, c_star: f32, h_t_sa: f32) {
//         match &mut self.n_s {
//             NumVisits::Exhausted => unreachable!(),
//             NumVisits::Count(n_s) => *n_s = n_s.checked_add(1).unwrap(),
//         }
//         let action_data = &mut self.actions[action_position];
//         if self.c_star > c_star {
//             self.c_star = c_star;
//             let next_position = action_data.next_position.as_mut().unwrap();
//             next_position.h_t_sa = h_t_sa;
//         }
//         //     // println!("improve node c_star!");
//         //     self.c_star = c_star;
//         //     let g = self.c - c_star;
//         //     debug_assert!(g > 0.0);
//         //     self.actions[action_position].update_g_sa(g);
//         // } else {
//         //     self.actions[action_position].decay(decay);
//         // }
//     }

//     // pub(crate) fn update_c_star_and_decay(
//     //     &mut self,
//     //     action_position: usize,
//     //     parent_c_star: f32,
//     //     decay: f32,
//     // ) {
//     //     let action_data = &mut self.actions[action_position];
//     //     let g_sa = self.c - parent_c_star;
//     //     if self.c_star > parent_c_star {
//     //         self.c_star = parent_c_star;
//     //         action_data.update_g_sa(g_sa);
//     //     } else {
//     //         action_data.decay(decay, g_sa);
//     //     }
//     // }

//     pub(crate) fn exhaust(&mut self, action_position: usize, c_star: f32) {
//         self.actions[action_position].next_position = None;
//         let foo = self.actions.iter().all(|a| a.next_position.is_none());
//         if foo {
//             self.n_s = NumVisits::Exhausted;
//         }
//         if self.c_star > c_star {
//             self.c_star = c_star;
//         }
//         // self.c_star = self.c_star.min(parent_c_star);
//         // let action_data = &mut self.actions[action_position];
//         // action_data.exhaust();
//     }

//     // pub(crate) fn exhaust_action(&mut self, action_position: usize) {
//     //     self.actions[action_position].exhaust();
//     // }
// }
