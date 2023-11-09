use std::collections::BTreeSet;

pub mod cost;
mod display;

#[derive(Clone, Debug)]
pub struct StateNode<S, A = usize> {
    pub(crate) state: S,
    pub(crate) time: usize,
    pub(crate) prohibited_actions: BTreeSet<A>,
}

impl<S, A> StateNode<S, A> {
    pub fn new(state: S, time: usize) -> Self {
        Self {
            state,
            time,
            prohibited_actions: BTreeSet::new(),
        }
    }

    pub fn state(&self) -> &S {
        &self.state
    }

    // pub fn reset(&mut self, time: usize) {
    //     self.time = time;
    //     self.prohibited_actions.clear();
    // }

    pub fn time(&self) -> usize {
        self.time
    }
}

pub trait Action<S> {
    fn index(&self) -> usize;
    unsafe fn from_index_unchecked(index: usize) -> Self;
}

pub trait State: Sized {
    type Actions: Action<Self>;
    fn actions(&self) -> impl Iterator<Item = Self::Actions>;
    fn is_terminal(&self) -> bool {
        self.actions().next().is_none()
    }
    unsafe fn act_unchecked(&mut self, action: &Self::Actions);
    fn act(&mut self, action: &Self::Actions)
    where
        Self::Actions: Eq + core::fmt::Display,
        Self: core::fmt::Display,
    {
        dbg!();
        debug_assert!(
            self.has_action(action),
            "action {} is not available in state {}",
            action,
            self,
        );
        dbg!();
        unsafe { self.act_unchecked(action) }
    }
    fn has_action(&self, action: &Self::Actions) -> bool
    where
        Self::Actions: Eq,
    {
        self.actions().any(|a| a.eq(action))
    }
}

impl<T: State> Action<StateNode<T>> for T::Actions {
    fn index(&self) -> usize {
        <Self as Action<T>>::index(self)
    }

    unsafe fn from_index_unchecked(index: usize) -> Self {
        <Self as Action<T>>::from_index_unchecked(index)
    }
}

pub trait ProhibitsActions {
    type Action;
    unsafe fn update_prohibited_actions_unchecked(
        &self,
        prohibited_actions: &mut BTreeSet<usize>,
        action: &Self::Action,
    );
}

impl<T> State for StateNode<T>
where
    T: State + ProhibitsActions<Action = <T as State>::Actions>,
    <T as State>::Actions: Action<T>,
{
    type Actions = T::Actions;

    fn actions(&self) -> impl Iterator<Item = Self::Actions> {
        if self.time == 0 {
            None
        } else {
            Some(self.state.actions().filter(|a| !self.prohibited_actions.contains(&a.index())))
        }.into_iter().flatten()
    }

    unsafe fn act_unchecked(&mut self, action: &Self::Actions) {
        // dbg!();
        let Self {
            state,
            time,
            prohibited_actions,
        } = self;
        // dbg!();
        state.act_unchecked(action);
        state.update_prohibited_actions_unchecked(prohibited_actions, action);
        *time -= 1;
    }

    fn is_terminal(&self) -> bool {
        self.actions().next().is_none()
    }

    fn act(&mut self, action: &Self::Actions)
    where
        Self::Actions: Eq + core::fmt::Display,
        Self: core::fmt::Display,
    {
        // dbg!();
        debug_assert!(
            self.has_action(action),
            "action {} is not available in state {}",
            action,
            self,
        );
        // dbg!();
        unsafe { self.act_unchecked(action) }
    }

    fn has_action(&self, action: &Self::Actions) -> bool
    where
        Self::Actions: Eq,
    {
        self.actions().any(|a| a.eq(action))
    }
}

pub trait Reset {
    fn reset(&mut self, time: usize);
}

impl<T> Reset for StateNode<T>
where
    T: State, // + Reset,
{
    fn reset(&mut self, time: usize) {
        // self.state.reset(time);
        self.time = time;
        self.prohibited_actions.clear();
    }
}

pub trait StateVec {
    const STATE_DIM: usize;
    const AVAILABLE_ACTIONS_BOOL_DIM: usize;
    fn write_vec(&self, vec: &mut [f32]) {
        let (state, actions) = vec.split_at_mut(Self::STATE_DIM);
        debug_assert_eq!(actions.len(), Self::AVAILABLE_ACTIONS_BOOL_DIM);
        self.write_vec_state_dims(state);
        self.write_vec_actions_dims(actions);
    }
    fn write_vec_state_dims(&self, state_vec: &mut [f32]);
    fn write_vec_actions_dims(&self, action_vec: &mut [f32]);
}

impl<T: StateVec> StateVec for StateNode<T> {
    const STATE_DIM: usize = T::STATE_DIM + 1;
    const AVAILABLE_ACTIONS_BOOL_DIM: usize = T::AVAILABLE_ACTIONS_BOOL_DIM;

    fn write_vec(&self, vec: &mut [f32]) {
        let (state, vec) = vec.split_at_mut(T::STATE_DIM);
        let (time, actions) = vec.split_at_mut(1);
        debug_assert_eq!(actions.len(), Self::AVAILABLE_ACTIONS_BOOL_DIM);
        self.write_vec_state_dims(state);
        time[0] = self.time as f32;
        self.write_vec_actions_dims(actions);
    }

    fn write_vec_state_dims(&self, state_vec: &mut [f32]) {
        self.state.write_vec_state_dims(state_vec)
    }

    fn write_vec_actions_dims(&self, action_vec: &mut [f32]) {
        self.state.write_vec_actions_dims(action_vec);
        self.prohibited_actions.iter().for_each(|&a| action_vec[a] = 0.);
    }
}
