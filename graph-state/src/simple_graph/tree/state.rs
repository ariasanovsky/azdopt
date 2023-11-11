use az_discrete_opt::state::{State, Action};

use super::PrueferCode;

pub struct PrueferCodeEntry {
    i: usize,
    parent: usize,
}

impl<const N: usize> State for PrueferCode<N> {
    type Actions = PrueferCodeEntry;

    fn actions(&self) -> impl Iterator<Item = Self::Actions> {
        self.code().into_iter().enumerate().map(|(i, parent)| {
            let before = 0..*parent;
            let after = *parent+1..N;
            before.chain(after).map(move |new_parent| PrueferCodeEntry {
                i,
                parent: new_parent,
            })
        }).flatten()
    }

    unsafe fn act_unchecked(&mut self, action: &Self::Actions) {
        let PrueferCodeEntry { i, parent } = action;
        self.code[*i] = *parent
    }
}

impl<const N: usize> Action<PrueferCode<N>> for PrueferCodeEntry {
    fn index(&self) -> usize {
        let PrueferCodeEntry { i, parent } = self;
        *i * N + *parent
    }

    unsafe fn from_index_unchecked(index: usize) -> Self {
        let i = index % N;
        let parent = index / N;
        Self { i, parent }
    }
}