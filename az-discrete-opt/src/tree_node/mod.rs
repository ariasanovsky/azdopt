// // use crate::{state::{State, Action}, path::ActionPathFor};

// // pub mod prohibitions;

// // pub trait TreeNode {
// //     type State;
// //     type Path;

// //     fn state(&self) -> &Self::State;
// //     fn path(&self) -> &Self::Path;
// //     fn apply_action<A>(&mut self, action: &A)
// //     where
// //         Self::State: State<Actions = A>,
// //         A: Action<Self::State> + PartialEq + Eq + core::fmt::Display,
// //     ;
// // }

// #[derive(Debug, Clone)]
// pub struct StatePath<S, P> {
//     state: S,
//     path: P,
// }

// impl<S, P> StatePath<S, P> {
//     pub fn new(state: S, path: P) -> Self {
//         Self { state, path }
//     }

//     pub fn state(&self) -> &S {
//         &self.state
//     }

//     pub fn path(&self) -> &P {
//         &self.path
//     }
// }

// impl<S, P> TreeNode for StatePath<S, P>
// where
//     S: core::fmt::Display,
//     P: ActionPathFor<S>,
// {
//     type State = S;

//     type Path = P;

//     fn state(&self) -> &Self::State {
//         &self.state
//     }

//     fn path(&self) -> &Self::Path {
//         &self.path
//     }

//     fn apply_action<A>(&mut self, action: &A)
//     where
//         Self::State: State<Actions = A>,
//         A: Action<Self::State> + PartialEq + Eq + core::fmt::Display,
//      {
//         self.state.act(action);
//         self.path.push(action);
//     }
// }

// impl<S, P> Action<StatePath<S, P>> for S::Actions
// where
//     S: State,
// {
//     fn index(&self) -> usize {
//         <S::Actions as Action<S>>::index(self)
//     }

//     unsafe fn from_index_unchecked(index: usize) -> Self {
//         <S::Actions as Action<S>>::from_index_unchecked(index)
//     }
// }

// impl<S, P> State for StatePath<S, P>
// where
//     S: State,
//     P: ActionPathFor<S>,
// {
//     type Actions = S::Actions;

//     fn actions(&self) -> impl Iterator<Item = Self::Actions> {
//         self.state.actions()
//     }

//     unsafe fn act_unchecked(&mut self, action: &Self::Actions) {
//         self.state.act_unchecked(action);
//         self.path.push(action);
//     }
// }