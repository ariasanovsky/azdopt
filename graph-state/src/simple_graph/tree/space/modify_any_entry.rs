use az_discrete_opt::space::StateActionSpace;

use crate::simple_graph::tree::PrueferCode;

use super::action::PrueferCodeEntry;

pub struct ModifyAnyPrueferCodeEntry<const N: usize>;

impl<const N: usize> StateActionSpace for ModifyAnyPrueferCodeEntry<N> {
    type State = PrueferCode<N>;

    type Action = PrueferCodeEntry;

    const DIM: usize = N * (N - 2);

    fn index(&self, action: &Self::Action) -> usize {
        action.action_index::<N>()
    }

    fn from_index(&self, index: usize) -> Self::Action {
        PrueferCodeEntry::from_action_index::<N>(index)
    }

    fn act(&self, state: &mut Self::State, action: &Self::Action) {
        let PrueferCodeEntry { i, parent } = action;
        state.modify_entry(*i, *parent);
    }

    fn action_indices(&self, state: &Self::State) -> impl Iterator<Item = usize> {
        state
            .code()
            .iter()
            .enumerate()
            .flat_map(|(i, p)| {
                let before = 0..*p;
                let after = *p + 1..N;
                before.chain(after).map(move |new_parent| PrueferCodeEntry {
                    i,
                    parent: new_parent,
                })
            })
            .map(|a| self.index(&a))
    }

    fn write_vec(&self, state: &Self::State, vec: &mut [f32]) {
        debug_assert!(vec.len() == Self::DIM);
        vec.fill(0.0);
        state.code().iter().enumerate().for_each(|(i, &parent)| {
            let entry = PrueferCodeEntry { i, parent };
            let index = self.index(&entry);
            vec[index] = 1.0;
        });
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        self.action_indices(state).next().is_none()
    }

    fn has_action(&self, state: &Self::State, action: &Self::Action) -> bool {
        let action_index = self.index(action);
        self.action_indices(state).any(|i| i == action_index)
    }
}
