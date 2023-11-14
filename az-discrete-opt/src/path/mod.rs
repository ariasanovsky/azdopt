use std::collections::{BTreeMap, BTreeSet};

use crate::state::{Action, State};

use self::never_repeat::ActionsNeverRepeat;

pub mod never_repeat;

/// If `S` and `A` are a state and action space, log the path of actions taken in a tree search.
/// Suppose we follow a sequence of actions `a_1, ..., a_n` from a root state `s_0` to end state `s_t`.
/// Then the path can be replayed to produce `s_t` from `s_0`.`
/// Depending on the states and actions, we may want to store the path in different ways.

pub trait ActionPath {
    /// Create a new path.
    fn new() -> Self;
    /// The number of actions in the path.
    fn len(&self) -> usize;
}

pub trait ActionPathFor<S>: ActionPath {
    /// Add an action to the path. Adding an action increments the depth of the path by 1.
    fn push(
        &mut self,
        action: &impl Action<S>,
    );
}

/// Sometimes, we need to preserve the precise order of actions taken.
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Debug)]
pub struct ActionSequence {
    actions: Vec<usize>,
}

impl ActionPath for ActionSequence {
    fn new() -> Self {
        Self { actions: vec![] }
    }
    fn len(&self) -> usize {
        self.actions.len()
    }
}

impl<S> ActionPathFor<S> for ActionSequence
where
    S: State,
{
    fn push(
        &mut self,
        action: &impl Action<S>,
    ) {
        self.actions.push(action.index());
    }
}

/// Sometimes, we need to preserve the precise order of actions taken, but may not repeat any action.
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Debug)]
pub struct OrderedActionSet {
    pub(crate) actions: Vec<usize>,
}

impl ActionPath for OrderedActionSet {
    fn new() -> Self {
        Self { actions: vec![] }
    }

    fn len(&self) -> usize {
        self.actions.len()
    }
}

impl<S> ActionPathFor<S> for OrderedActionSet
where
    S: State + ActionsNeverRepeat<S::Actions>,
{
    fn push(
        &mut self,
        action: &impl Action<S>,
    ) {
        debug_assert!(
            !self.actions.contains(&action.index()),
            "Action {} was already in the path.\nAre you sure this state space never repeats actions?",
            action.index()
        );
        self.actions.push(action.index());
    }
}

/// Sometimes actions can be repeated, but the actions may be ignored.
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Debug)]
pub struct ActionMultiset {
    pub(crate) actions: BTreeMap<usize, usize>,
}

impl ActionPath for ActionMultiset {
    fn new() -> Self {
        Self { actions: BTreeMap::new() }
    }

    fn len(&self) -> usize {
        self.actions.values().sum()
    }
}

// impl<S> ActionPathFor<S> for ActionMultiset {
//     fn push(
//         &mut self,
//         action: &impl Action<S>,
//     ) {
//         self.actions.entry(action.index()).and_modify(|count| *count += 1).or_insert(1);
//     }
// }

/// Sometimes, actions may not be repeated, so we can use a set.
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Debug)]
pub struct ActionSet {
    pub(crate) actions: BTreeSet<usize>,
}

impl ActionPath for ActionSet {
    fn new() -> Self {
        Self { actions: BTreeSet::new() }
    }

    fn len(&self) -> usize {
        self.actions.len()
    }
}

// impl<S> ActionPathFor<S> for ActionSet {
//     fn push(
//         &mut self,
//         action: &impl Action<S>,
//     ) {
//         let _inserted = self.actions.insert(action.index());
//         debug_assert!(
//             _inserted,
//             "action {} was already in the path",
//             action.index()
//         );
//     }
// }