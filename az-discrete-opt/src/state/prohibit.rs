use std::collections::BTreeSet;

#[derive(Clone, Debug)]
pub struct WithProhibitions<S> {
    pub state: S,
    pub prohibited_actions: BTreeSet<usize>,
}

impl<S> WithProhibitions<S> {
    pub fn new(state: S) -> Self {
        Self {
            state,
            prohibited_actions: BTreeSet::new(),
        }
    }

    pub fn extend_prohibitions(&mut self, prohibited_actions: impl IntoIterator<Item = usize>) {
        self.prohibited_actions.extend(prohibited_actions);
    }
}
