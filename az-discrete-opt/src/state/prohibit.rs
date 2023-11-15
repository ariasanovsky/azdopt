use std::collections::BTreeSet;

pub struct WithProhibitions<S> {
    pub state: S,
    pub prohibited_actions: BTreeSet<usize>,
}

// impl<S> WithProhibitions<S> {
//     pub fn new(state: S) -> Self {
//         Self {
//             state,
//             prohibited_actions: BTreeSet::new(),
//         }
//     }
// }
