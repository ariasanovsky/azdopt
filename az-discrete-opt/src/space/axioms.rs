use crate::path::{
    multiset::ActionMultiset, ord_set::OrderedActionSet, set::ActionSet, ActionPathFor,
};

use super::StateActionSpace;

/// # Safety
/// `Self` must have the state and action space restrictions that prevent actions from being repeated
pub unsafe trait ActionsNeverRepeat: StateActionSpace {}
/// # Safety
/// `Self` must have the state and action space restrictions that allow us to forget the order of actions
pub unsafe trait ActionOrderIndependent: StateActionSpace {}

unsafe impl<Space> ActionPathFor<Space> for OrderedActionSet where
    Space: StateActionSpace + ActionsNeverRepeat
{
}

unsafe impl<Space> ActionPathFor<Space> for ActionMultiset where
    Space: StateActionSpace + ActionOrderIndependent
{
}

unsafe impl<Space> ActionPathFor<Space> for ActionSet where
    Space: StateActionSpace + ActionsNeverRepeat + ActionOrderIndependent
{
}
