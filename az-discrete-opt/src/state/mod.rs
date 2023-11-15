pub mod cost;
pub mod prohibit;
pub mod vec;

// #[derive(Clone, Debug)]
// pub struct StateNode<S, A = usize> {
//     pub(crate) state: S,
//     pub(crate) time: usize,
//     pub(crate) prohibited_actions: BTreeSet<A>,
// }

// impl<S, A> StateNode<S, A> {
//     pub fn new(state: S, time: usize) -> Self {
//         Self {
//             state,
//             time,
//             prohibited_actions: BTreeSet::new(),
//         }
//     }

//     pub fn state(&self) -> &S {
//         &self.state
//     }

//     // pub fn reset(&mut self, time: usize) {
//     //     self.time = time;
//     //     self.prohibited_actions.clear();
//     // }

//     pub fn time(&self) -> usize {
//         self.time
//     }
// }

// pub trait Action<S, Space = S>: Sized {
//     const DIM: usize;
//     fn index(&self) -> usize;
//     fn from_index(index: usize) -> Self;
//     fn act(&self, state: &mut S);
//     fn actions(state: &S) -> impl Iterator<Item = usize>;
//     fn is_terminal(state: &S) -> bool {
//         Self::actions(state).next().is_none()
//     }
// }

// pub trait State: Sized {
//     fn index<A>(a: &A) -> usize
//     where
//         A: Action<Self>,
//     {
//         A::index(a)
//     }

//     fn from_index<A>(index: usize) -> A
//     where
//         A: Action<Self>,
//     {
//         A::from_index(index)
//     }

//     fn act<A>(&mut self, a: &A)
//     where
//         A: Action<Self>,
//     {
//         A::act(a, self)
//     }

//     fn actions<A>(&self) -> impl Iterator<Item = usize>
//     where
//         A: Action<Self>
//     {
//         A::actions(self)
//     }

//     fn is_terminal<A>(&self) -> bool
//     where
//         A: Action<Self>,
//     {
//         A::is_terminal(self)
//     }
// }

// impl<S> State for S {}

// pub trait TreeNode {
//     type Action;
//     type State;
//     type Path;
//     fn state(&self) -> &Self::State;
//     fn path(&self) -> &Self::Path;
// }

// pub trait State: Sized {
//     type Actions: Action<Self>;
//     fn actions(&self) -> impl Iterator<Item = Self::Actions>;
//     fn is_terminal(&self) -> bool {
//         self.actions().next().is_none()
//     }
//     /// # Safety
//     /// This function should only be called with actions that are available in the state.
//     /// It is used to generated a default implementation of `act` which does perform a check in debug mode.
//     unsafe fn act_unchecked(&mut self, action: &Self::Actions);
//     fn act(&mut self, action: &Self::Actions)
//     where
//         Self::Actions: Eq + core::fmt::Display,
//         Self: core::fmt::Display,
//     {
//         debug_assert!(
//             self.has_action(action),
//             "action {} is not available in state {}",
//             action,
//             self,
//         );
//         unsafe { self.act_unchecked(action) }
//     }
//     fn has_action(&self, action: &Self::Actions) -> bool
//     where
//         Self::Actions: Eq,
//     {
//         self.actions().any(|a| a.eq(action))
//     }
// }

// impl<T: State> Action<StateNode<T>> for T::Actions {
//     fn index(&self) -> usize {
//         <Self as Action<T>>::index(self)
//     }

//     unsafe fn from_index_unchecked(index: usize) -> Self {
//         <Self as Action<T>>::from_index_unchecked(index)
//     }
// }

// pub trait ProhibitsActions<A> {
//     /// # Safety
//     /// This function should only be called with actions that are available in the state.
//     /// It does not perform checks and assumes the caller has already checked that the action is available.
//     unsafe fn update_prohibited_actions_unchecked(
//         &self,
//         prohibited_actions: &mut BTreeSet<usize>,
//         action: &A,
//     );
// }

// pub struct ExcludingProhibitedActions<S> {
//     pub state: S,
//     pub prohibited_actions: BTreeSet<usize>,
// }

// impl<S: State> Action<ExcludingProhibitedActions<S>> for S::Actions {
//     fn index(&self) -> usize {
//         <Self as Action<S>>::index(self)
//     }

//     unsafe fn from_index_unchecked(index: usize) -> Self {
//         <Self as Action<S>>::from_index_unchecked(index)
//     }
// }

// impl<S> State for ExcludingProhibitedActions<S>
// where
//     S: State + ProhibitsActions<S::Actions>,
//     S::Actions: Action<S>,
// {
//     type Actions = S::Actions;

//     fn actions(&self) -> impl Iterator<Item = Self::Actions> {
//         let Self { state, prohibited_actions } = self;
//         state.actions().filter(move |a| !prohibited_actions.contains(&a.index()))
//     }

//     unsafe fn act_unchecked(&mut self, action: &Self::Actions) {
//         let Self { state, prohibited_actions } = self;
//         debug_assert!(
//             !prohibited_actions.contains(&action.index()),
//             "action {} is prohibited",
//             action.index(),
//         );
//         state.act_unchecked(action);
//     }
// }

// impl<T> State for StateNode<T>
// where
//     T: State + ProhibitsActions<T::Actions>,
//     T::Actions: Action<T>,
// {
//     type Actions = T::Actions;

//     fn actions(&self) -> impl Iterator<Item = Self::Actions> {
//         if self.time == 0 {
//             None
//         } else {
//             Some(
//                 self.state
//                     .actions()
//                     .filter(|a| !self.prohibited_actions.contains(&a.index())),
//             )
//         }
//         .into_iter()
//         .flatten()
//     }

//     unsafe fn act_unchecked(&mut self, action: &Self::Actions) {
//         // dbg!();
//         let Self {
//             state,
//             time,
//             prohibited_actions,
//         } = self;
//         // dbg!();
//         state.act_unchecked(action);
//         state.update_prohibited_actions_unchecked(prohibited_actions, action);
//         *time -= 1;
//     }

//     fn is_terminal(&self) -> bool {
//         self.actions().next().is_none()
//     }

//     fn act(&mut self, action: &Self::Actions)
//     where
//         Self::Actions: Eq + core::fmt::Display,
//         Self: core::fmt::Display,
//     {
//         debug_assert!(
//             self.has_action(action),
//             "action {} is not available in state {}",
//             action,
//             self,
//         );
//         unsafe { self.act_unchecked(action) }
//     }

//     fn has_action(&self, action: &Self::Actions) -> bool
//     where
//         Self::Actions: Eq,
//     {
//         self.actions().any(|a| a.eq(action))
//     }
// }

// pub trait Reset {
//     fn reset(&mut self, time: usize);
// }

// impl<T> Reset for StateNode<T>
// where
//     T: State, // + Reset,
// {
//     fn reset(&mut self, time: usize) {
//         // self.state.reset(time);
//         self.time = time;
//         self.prohibited_actions.clear();
//     }
// }

// // pub trait StateVec {
// //     const STATE_DIM: usize;
// //     const ACTION_DIM: usize;
// //     fn write_vec(&self, vec: &mut [f32]) {
// //         let (state, actions) = vec.split_at_mut(Self::STATE_DIM);
// //         if Self::WRITE_ACTION_DIMS {
// //             debug_assert_eq!(actions.len(), Self::ACTION_DIM);
// //         } else {
// //             debug_assert_eq!(actions.len(), 0);
// //         }
// //         self.write_vec_state_dims(state);
// //         self.write_vec_actions_dims(actions);
// //     }
// //     fn write_vec_state_dims(&self, state_vec: &mut [f32]);
// //     fn write_vec_actions_dims(&self, action_vec: &mut [f32]);
// // }

// // impl<T: StateVec> StateVec for StateNode<T> {
// //     const STATE_DIM: usize = T::STATE_DIM + 1;
// //     const ACTION_DIM: usize = T::ACTION_DIM;
// //     const WRITE_ACTION_DIMS: bool = T::WRITE_ACTION_DIMS;
// //     fn write_vec(&self, vec: &mut [f32]) {
// //         todo!("break up `StateNode`");
// //         // one wrapper for time
// //         // the other for prohibited actions
// //         let (state_with_time, actions) = vec.split_at_mut(Self::STATE_DIM);
// //         if Self::WRITE_ACTION_DIMS {
// //             debug_assert_eq!(actions.len(), Self::ACTION_DIM);
// //             self.write_vec_state_dims(state_with_time);
// //         } else {
// //             debug_assert_eq!(actions.len(), 0);
// //             self.write_vec_state_dims(state_with_time);
// //             self.write_vec_actions_dims(actions);
// //         }
// //     }

// //     fn write_vec_state_dims(&self, state_vec: &mut [f32]) {
// //         let (time, state) = state_vec.split_at_mut(1);
// //         time[0] = self.time as f32;
// //         self.state.write_vec_state_dims(state_vec)
// //     }

// //     fn write_vec_actions_dims(&self, action_vec: &mut [f32]) {
// //         self.state.write_vec_actions_dims(action_vec);
// //         self.prohibited_actions
// //             .iter()
// //             .for_each(|&a| action_vec[a] = 0.);
// //     }
// // }
