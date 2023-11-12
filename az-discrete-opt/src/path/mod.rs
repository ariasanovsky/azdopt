use std::collections::{BTreeMap, BTreeSet};

use crate::state::Action;

/// If `S` and `A` are a state and action space, log the path of actions taken in a tree search.
/// Suppose we follow a sequence of actions `a_1, ..., a_n` from a root state `s_0` to end state `s_t`.
/// Then the path can be replayed to produce `s_t` from `s_0`.`
/// Depending on the states and actions, we may want to store the path in different ways.
pub trait ActionPath<S> {
    /// Create a new path.
    fn new(action_1: &impl Action<S>) -> Self;
    /// Add an action to the path. Adding an action increments the depth of the path by 1.
    fn push(
        &mut self,
        action: &impl Action<S>,
    );
    /// The number of actions in the path.
    fn len(&self) -> usize;
}

/// Sometimes, we need to preserve the precise order of actions taken.
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Debug)]
pub struct ActionSequence {
    actions: Vec<usize>,
}

impl<S> ActionPath<S> for ActionSequence {
    fn new(action_1: &impl Action<S>) -> Self {
        Self {
            actions: vec![action_1.index()]
        }
    }
    fn push(
        &mut self,
        action: &impl Action<S>,
    ) {
        self.actions.push(action.index());
    }

    fn len(&self) -> usize {
        self.actions.len()
    }
}

/// Sometimes actions can be sorted into a canonical order.
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Debug)]
pub struct ActionMultiset {
    actions: BTreeMap<usize, usize>,
}

impl<S> ActionPath<S> for ActionMultiset {
    fn new(action_1: &impl Action<S>) -> Self {
        Self {
            actions: BTreeMap::from([(action_1.index(), 1)])
        }
    }
    fn push(
        &mut self,
        action: &impl Action<S>,
    ) {
        self.actions.entry(action.index()).and_modify(|count| *count += 1).or_insert(1);
    }

    fn len(&self) -> usize {
        self.actions.iter().map(|(_, count)| count).sum()
    }
}

/// Sometimes, actions may not be repeated, so we can use a set.
#[derive(Clone, PartialEq, PartialOrd, Eq, Ord, Debug)]
pub struct ActionSet {
    actions: BTreeSet<usize>,
}

impl<S> ActionPath<S> for ActionSet {
    fn new(action_1: &impl Action<S>) -> Self {
        todo!()
    }
    fn push(
        &mut self,
        action: &impl Action<S>,
    ) {
        let _inserted = self.actions.insert(action.index());
        debug_assert!(
            _inserted,
            "action {} was already in the path",
            action.index()
        );
    }

    fn len(&self) -> usize {
        self.actions.len()
    }
}