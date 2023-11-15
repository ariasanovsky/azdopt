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
        let i = 
            (0..N)
            .map(|p| PrueferCodeEntry { i: *i, parent: p })
            .map(|a| Self::index(&a));
        state.prohibited_actions.extend(i)
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

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use az_discrete_opt::space::StateActionSpace;

    use crate::simple_graph::tree::{PrueferCode, space::modify_each_entry_once::ModifyEachPrueferCodeEntriesExactlyOnce};

    type SASpace<const N: usize> = ModifyEachPrueferCodeEntriesExactlyOnce<N>;
    type State<const N: usize> = <ModifyEachPrueferCodeEntriesExactlyOnce<N> as StateActionSpace>::State;
    type Action<const N: usize> = <ModifyEachPrueferCodeEntriesExactlyOnce<N> as StateActionSpace>::Action;
    
    #[test]
    fn after_modifying_a_pruefer_code_entry_the_entry_can_no_longer_be_modified() {
        type Space = SASpace<4>;
        type S = State<4>;
        type A = Action<4>;
        let mut code = State {
            state: PrueferCode { code: [0, 0, 0, 0] },
            prohibited_actions: BTreeSet::from([
                // Action { i: 0, parent: 0 },
                Action { i: 0, parent: 1 },
                // Action { i: 0, parent: 2 },
                // Action { i: 0, parent: 3 },
                // Action { i: 1, parent: 0 },
                // Action { i: 1, parent: 1 },
                // Action { i: 1, parent: 2 },
                Action { i: 1, parent: 3 },
            ].map(|a| Space::index(&a))),
        };
        let actions_to_take = [
            Action { i: 0, parent: 1 },
            Action { i: 1, parent: 3 },
        ];
        let action_sets: [BTreeSet<A>; 3] = [
            BTreeSet::from([
                Action { i: 0, parent: 0 },
                // Action { i: 0, parent: 1 },
                Action { i: 0, parent: 2 },
                Action { i: 0, parent: 3 },
                Action { i: 1, parent: 0 },
                Action { i: 1, parent: 1 },
                Action { i: 1, parent: 2 },
                // Action { i: 1, parent: 3 },
            ]),
            BTreeSet::from([
                Action { i: 1, parent: 0 },
                Action { i: 1, parent: 1 },
                Action { i: 1, parent: 2 },
                // Action { i: 1, parent: 3 },
            ]),
            BTreeSet::from([]),
        ];
        // test the action set before taking actions
        let actions = Space::actions(&code).map(|i| Space::from_index(i)).collect::<BTreeSet<_>>();
        assert_eq!(actions, action_sets[0]);
        todo!();
    }
}