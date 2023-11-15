use az_discrete_opt::{space::{StateActionSpace, axioms::{ActionsNeverRepeat, ActionOrderIndependent}}, state::prohibit::WithProhibitions};

use crate::simple_graph::tree::{state::PrueferCodeEntry, PrueferCode};

pub struct ModifyEachPrueferCodeEntriesExactlyOnce<const N: usize>;

unsafe impl<const N: usize> ActionsNeverRepeat for ModifyEachPrueferCodeEntriesExactlyOnce<N> {}
unsafe impl<const N: usize> ActionOrderIndependent for ModifyEachPrueferCodeEntriesExactlyOnce<N> {}

impl<const N: usize> StateActionSpace for ModifyEachPrueferCodeEntriesExactlyOnce<N> {
    type State = WithProhibitions<PrueferCode<N>>;

    type Action = PrueferCodeEntry;

    const DIM: usize = 2 * N * (N - 2);

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
        state.state.code[*i] = *parent;
        todo!();
        // state.prohibited_actions.extend()
    }

    fn actions(state: &Self::State) -> impl Iterator<Item = usize> {
        state.state.code().iter().enumerate().map(|(i, p)| {
            let before = 0..*p;
            let after = *p+1..N;
            before.chain(after).map(move |new_parent| PrueferCodeEntry {
                i,
                parent: new_parent,
            })
        }).flatten().map(|a| Self::index(&a))
        .filter(move |&i| !state.prohibited_actions.contains(&i))
    }

    fn write_vec(state: &Self::State, vec: &mut [f32]) {
        debug_assert!(vec.len() == Self::DIM);
        vec.fill(0.0);
        let (state_vec, action_vec) = vec.split_at_mut(N * (N - 2));
        state.state.code().iter().enumerate().for_each(|(i, &parent)| {
            let entry = PrueferCodeEntry { i, parent };
            let index = Self::index(&entry);
            state_vec[index] = 1.0;
        });
        action_vec.iter_mut().enumerate().for_each(|(i, a)| {
            if state.prohibited_actions.contains(&i) {
                *a = 1.0;
            }
        })
    }

    fn is_terminal(state: &Self::State) -> bool {
        Self::actions(state).next().is_none()
    }

    fn has_action(state: &Self::State, action: &Self::Action) -> bool {
        let action_index = Self::index(action);
        Self::actions(state).any(|i| i == action_index)
    }
}
