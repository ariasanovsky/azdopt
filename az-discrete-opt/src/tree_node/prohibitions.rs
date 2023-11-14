use std::collections::BTreeSet;

use crate::{path::{ActionPathFor, OrderedActionSet, ActionMultiset, ActionSet}, state::{State, Action}};

#[derive(Clone, Debug)]
pub struct WithActionProhibitions<S> {
    pub(crate) state: S,
    pub(crate) prohibited_actions: BTreeSet<usize>,
}

// impl<S: State> State for WithActionProhibitions<S> {
//     type Actions = S::Actions;

//     fn actions(&self) -> impl Iterator<Item = Self::Actions> {
//         todo!()
//     }

//     unsafe fn act_unchecked(&mut self, action: &Self::Actions) {
//         todo!()
//     }
// }

impl<S> WithActionProhibitions<S> {
    pub fn new(state: S) -> Self {
        Self {
            state,
            prohibited_actions: BTreeSet::new(),
        }
    }

    pub fn state(&self) -> &S {
        &self.state
    }

    pub fn prohibited_actions(&self) -> &BTreeSet<usize> {
        &self.prohibited_actions
    }
}

/// Suppose `S` is a state space with action space `A`.
/// We use this trait to define a restricted action space for `WithActionProhibitions<S>`.
pub trait ProhibitActionsFor<A, P> {
    fn update_prohibited_actions(
        &self,
        prohibited_actions: &mut BTreeSet<usize>,
        action: &A,
    );
}

/// We can force every path in the search space to be an ordered set by prohibiting every action we take.
/// This is useful to prevent small cycles when actions can be reversed.
impl<S> ActionPathFor<WithActionProhibitions<S>> for OrderedActionSet {
    fn push(
        &mut self,
        action: &impl Action<WithActionProhibitions<S>>,
    ) {
        self.actions.push(action.index());
    }
}

/// We may be able to restrict the action space so that search paths can be treated as unordered multisets of actions.
/// This identifies nodes in the search tree which correspond to the same state.
impl<S> ActionPathFor<WithActionProhibitions<S>> for ActionMultiset
where
    S: State + ProhibitActionsFor<S::Actions, Self>,
{
    fn push(
        &mut self,
        action: &impl Action<WithActionProhibitions<S>>,
    ) {
        self.actions.entry(action.index()).and_modify(|count| *count += 1).or_insert(1);
    }
}

/// We may even be able to restrict the action space so that search paths can be treated as unordered sets of actions.
/// This identifies even more nodes in the search tree which correspond to the same state.
impl<S> ActionPathFor<WithActionProhibitions<S>> for ActionSet
where
    S: State + ProhibitActionsFor<S::Actions, Self>,
{
    fn push(
        &mut self,
        action: &impl Action<WithActionProhibitions<S>>,
    ) {
        let inserted = self.actions.insert(action.index());
        debug_assert!(
            inserted,
            "Action {} was already in the path.\nAre you sure this state space never repeats actions?",
            action.index()
        );
    }
}