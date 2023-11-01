use std::collections::BTreeSet;

pub trait Cost {
    fn cost(&self) -> f32;
}

#[derive(Clone)]
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
        Self::Actions: Eq,
    {
        assert!(self.has_action(action));
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
}

impl<T: State> State for StateNode<T>
where
    <T as State>::Actions: Action<T>,
{
    type Actions = T::Actions;

    fn actions(&self) -> impl Iterator<Item = Self::Actions> {
        self.state.actions().filter(|a| !self.prohibited_actions.contains(&a.index()))
    }

    unsafe fn act_unchecked(&mut self, action: &Self::Actions) {
        todo!()
    }
}

pub trait Reset {
    fn reset(&mut self, time: usize);
}
