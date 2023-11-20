use std::collections::BTreeSet;

use crate::log::ShortForm;

#[derive(Clone, Debug)]
pub struct WithProhibitions<S> {
    pub state: S,
    pub prohibited_actions: BTreeSet<usize>,
}

impl<S> WithProhibitions<S> {
    pub fn new(state: S, prohibited_actions: impl IntoIterator<Item = usize>) -> Self {
        Self {
            state,
            prohibited_actions: BTreeSet::from_iter(prohibited_actions),
        }
    }

    pub fn extend_prohibitions(&mut self, prohibited_actions: impl IntoIterator<Item = usize>) {
        self.prohibited_actions.extend(prohibited_actions);
    }
}

// todo! derive? include prohibited actions?
impl<S: ShortForm> ShortForm for WithProhibitions<S> {
    fn short_form(&self) -> String {
        self.state.short_form()
    }
}
