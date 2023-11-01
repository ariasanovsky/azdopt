use core::fmt::Display;

use super::StateNode;

impl<S: Display> Display for StateNode<S> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            state,
            time,
            prohibited_actions,
        } = self;
        write!(f, "state =\n{state}\ntime = {time}\nprohibited_actions = {prohibited_actions:?}")
    }
}