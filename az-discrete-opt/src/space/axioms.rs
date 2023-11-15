use super::StateActionSpace;

pub unsafe trait ActionsNeverRepeat: StateActionSpace {}
pub unsafe trait ActionOrderIndependent: StateActionSpace {}
