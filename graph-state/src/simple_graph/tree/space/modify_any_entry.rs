use az_discrete_opt::space::StateActionSpace;

use crate::simple_graph::tree::{PrueferCode, state::PrueferCodeEntry};

struct ModifyAnyPrueferCodeEntry<const N: usize>;

impl<const N: usize> StateActionSpace for ModifyAnyPrueferCodeEntry<N> {
    type State = PrueferCode<N>;

    type Action = PrueferCodeEntry;

    const DIM: usize = N * (N - 2);

    fn index(action: &Self::Action) -> usize {
        let PrueferCodeEntry { i, parent } = action;
        *i * N + *parent
    }

    fn from_index(index: usize) -> Self::Action {
        let i = index / N;
        let parent = index % N;
        PrueferCodeEntry { i, parent }
    }

    fn act(state: &mut Self::State, action: &Self::Action) {
        let PrueferCodeEntry { i, parent } = action;
        state.code[*i] = *parent
    }

    fn actions(state: &Self::State) -> impl Iterator<Item = usize> {
        state.code().iter().enumerate().map(|(i, p)| {
            let before = 0..*p;
            let after = *p+1..N;
            before.chain(after).map(move |new_parent| PrueferCodeEntry {
                i,
                parent: new_parent,
            })
        }).flatten().map(|a| Self::index(&a))
    }

    fn write_vec(state: &Self::State, vec: &mut [f32]) {
        debug_assert!(vec.len() == Self::DIM);
        vec.fill(0.0);
        state.code().iter().enumerate().for_each(|(i, &parent)| {
            let entry = PrueferCodeEntry { i, parent };
            let index = Self::index(&entry);
            vec[index] = 1.0;
        });
    }

    fn is_terminal(state: &Self::State) -> bool {
        Self::actions(state).next().is_none()
    }

    fn has_action(state: &Self::State, action: &Self::Action) -> bool {
        let action_index = Self::index(action);
        Self::actions(state).any(|i| i == action_index)
    }
}