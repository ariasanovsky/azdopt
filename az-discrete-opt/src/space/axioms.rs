use crate::path::{
    multiset::ActionMultiset, ord_set::OrderedActionSet, set::ActionSet, ActionPathFor,
};

/// # Safety
/// `Self` must have the state and action space restrictions that prevent actions from being repeated
pub unsafe trait ActionsNeverRepeat {}
/// # Safety
/// `Self` must have the state and action space restrictions that allow us to forget the order of actions
pub unsafe trait ActionOrderIndependent {}

unsafe impl<Space> ActionPathFor<Space> for OrderedActionSet where
    Space: ActionsNeverRepeat
{
}

unsafe impl<Space> ActionPathFor<Space> for ActionMultiset where
    Space: ActionOrderIndependent
{
}

unsafe impl<Space> ActionPathFor<Space> for ActionSet where
    Space: ActionsNeverRepeat + ActionOrderIndependent
{
}
