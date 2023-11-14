
use crate::{path::{never_repeat::ActionsNeverRepeat, OrderedActionSet, ActionPathFor, ActionPath}, tree_node::TreeNode, log::ShortForm};

use super::{Action, State};

// todo! delete this
#[derive(Debug, Clone)]
pub struct ExcludePreviousActions<S> {
    state: S,
    previous_actions: OrderedActionSet,
}

impl<S> ExcludePreviousActions<S> {
    pub fn new(state: S) -> Self {
        Self {
            state,
            previous_actions: OrderedActionSet::new(),
        }
    }
}

impl<S: State> Action<ExcludePreviousActions<S>> for S::Actions
{
    fn index(&self) -> usize {
        <Self as Action<S>>::index(self)
    }

    unsafe fn from_index_unchecked(index: usize) -> Self {
        <Self as Action<S>>::from_index_unchecked(index)
    }
}


impl<S: State> ActionsNeverRepeat<S::Actions> for ExcludePreviousActions<S>
where
    S: State,
{}

impl<S> State for ExcludePreviousActions<S>
where
    S: State,
    S::Actions: Action<S>,
{
    type Actions = S::Actions;

    fn actions(&self) -> impl Iterator<Item = Self::Actions> {
        let Self { state, previous_actions: actions } = self;
        state.actions().filter(move |a| !actions.actions.contains(&a.index()))
    }

    unsafe fn act_unchecked(&mut self, action: &Self::Actions) {
        let Self { state, previous_actions } = self;
        state.act_unchecked(action);
        // todo! traits
        previous_actions.actions.push(action.index());
    }
}

impl<S> TreeNode for ExcludePreviousActions<S> {
    type State = S;

    type Path = OrderedActionSet;

    fn state(&self) -> &Self::State {
        &self.state
    }

    fn path(&self) -> &Self::Path {
        &self.previous_actions
    }

    fn apply_action<A>(&mut self, action: &A)
    where
        Self::State: State<Actions = A>,
        A: Action<Self::State>,
    {
        let Self { state, previous_actions } = self;
        unsafe { state.act_unchecked(action) };
        previous_actions.actions.push(action.index());
    }
}

// impl<S> StateVec for ExcludePreviousActions<S>
// where
//     S: StateVec,
// {
//     const STATE_DIM: usize = S::STATE_DIM;

//     const ACTION_DIM: usize = S::ACTION_DIM;
    
//     const WRITE_ACTION_DIMS: bool = true;

//     fn write_vec_state_dims(&self, state_vec: &mut [f32]) {
//         self.state.write_vec_state_dims(state_vec)
//     }

//     fn write_vec_actions_dims(&self, action_vec: &mut [f32]) {
//         self.state.write_vec_actions_dims(action_vec);
//         self.previous_actions
//             .actions
//             .iter()
//             .for_each(|&a| action_vec[a] = 0.);
//     }

//     fn write_vec(&self, vec: &mut [f32]) {
//         dbg!(
//             vec.len(),
//             Self::STATE_DIM,
//             Self::ACTION_DIM,
//             Self::WRITE_ACTION_DIMS,
//         );
//         let (state, actions) = vec.split_at_mut(Self::STATE_DIM);
//         debug_assert_eq!(actions.len(), Self::ACTION_DIM);
//         self.write_vec_state_dims(state);
//         self.write_vec_actions_dims(actions);
//     }
// }

impl<S: ShortForm> ShortForm for ExcludePreviousActions<S> {
    fn short_form(&self) -> String {
        let Self { state, previous_actions } = self;
        todo!()
    }
}
//     const STATE_DIM: usize = S::STATE_DIM;

//     const AVAILABLE_ACTIONS_BOOL_DIM: usize = S::AVAILABLE_ACTIONS_BOOL_DIM;


//     fn write_vec(&self, vec: &mut [f32]) {
//         let (state, actions) = vec.split_at_mut(Self::STATE_DIM);
//         debug_assert_eq!(actions.len(), Self::AVAILABLE_ACTIONS_BOOL_DIM);
//         self.write_vec_state_dims(state);
//         self.write_vec_actions_dims(actions);
//     }

//     fn write_vec_state_dims(&self, state_vec: &mut [f32]) {
//         self.state.write_vec_state_dims(state_vec)

//     }

//     fn write_vec_actions_dims(&self, action_vec: &mut [f32]) {
//         self.state.write_vec_actions_dims(action_vec);
//         self.previous_actions
//             .actions
//             .iter()
//             .for_each(|&a| action_vec[a] = 0.);
//     }
// }

// impl<S> ActionPath<S> for OrderedActionSet
// where
//     S: State + ActionsNeverRepeat<S::Actions>,
// {
//     fn new(action_1: &impl Action<S>) -> Self {
//         Self {
//             actions: vec![action_1.index()],
//         }
//     }
//     fn push(
//         &mut self,
//         action: &impl Action<S>,
//     ) {
//         debug_assert!(
//             !self.actions.contains(&action.index()),
//             "action {} was already in the path",
//             action.index()
//         );
//         self.actions.push(action.index());
//     }

//     fn len(&self) -> usize {
//         self.actions.len()
//     }
// }

// impl<S> Action<ExcludePreviousActions<S>> for S::Actions
// where
//     S: State,
// {
//     fn index(&self) -> usize {
//         <Self as Action<S>>::index(self)
//     }

//     unsafe fn from_index_unchecked(index: usize) -> Self {
//         <Self as Action<S>>::from_index_unchecked(index)
//     }
// }

// impl<S> State for ExcludePreviousActions<S>
// where
//     S: State,
//     S::Actions: Action<S>,
// {
//     type Actions = <S as State>::Actions;

//     fn actions(&self) -> impl Iterator<Item = Self::Actions> {
//         let Self { state, previous_actions: actions } = self;
//         state.actions()
//     }

//     unsafe fn act_unchecked(&mut self, action: &Self::Actions) {
//         let Self { state, previous_actions } = self;
//         state.act_unchecked(action);
//         previous_actions.push(action);
//     }
// }