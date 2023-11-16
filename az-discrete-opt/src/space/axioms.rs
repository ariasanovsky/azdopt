use crate::path::{ActionPathFor, ord_set::OrderedActionSet, multiset::ActionMultiset, set::ActionSet};

use super::StateActionSpace;

pub unsafe trait ActionsNeverRepeat: StateActionSpace {}
pub unsafe trait ActionOrderIndependent: StateActionSpace {}

unsafe impl<Space> ActionPathFor<Space> for OrderedActionSet
where
    Space: StateActionSpace + ActionsNeverRepeat
{}

unsafe impl<Space> ActionPathFor<Space> for ActionMultiset
where
    Space: StateActionSpace + ActionOrderIndependent
{}

unsafe impl<Space> ActionPathFor<Space> for ActionSet
where
    Space: StateActionSpace + ActionsNeverRepeat + ActionOrderIndependent
{}