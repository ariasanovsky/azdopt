use az_discrete_opt::{
    space::{
        axioms::{ActionOrderIndependent, ActionsNeverRepeat},
        StateActionSpace,
    },
    state::prohibit::WithProhibitions,
};

use crate::simple_graph::tree::PrueferCode;

use super::action::PrueferCodeEntry;

pub struct ModifyEachPrueferCodeEntriesExactlyOnce<const N: usize>;

unsafe impl<const N: usize> ActionsNeverRepeat for ModifyEachPrueferCodeEntriesExactlyOnce<N> {}
unsafe impl<const N: usize> ActionOrderIndependent for ModifyEachPrueferCodeEntriesExactlyOnce<N> {}

impl<const N: usize> StateActionSpace for ModifyEachPrueferCodeEntriesExactlyOnce<N> {
    type State = WithProhibitions<PrueferCode<N>>;

    type Action = PrueferCodeEntry;

    const DIM: usize = 2 * N * (N - 2);

    fn index(&self, action: &Self::Action) -> usize {
        action.action_index::<N>()
    }

    fn from_index(&self, index: usize) -> Self::Action {
        let i = index / N;
        let parent = index % N;
        PrueferCodeEntry { i, parent }
    }

    fn act(&self, state: &mut Self::State, action: &Self::Action) {
        let PrueferCodeEntry { i, parent } = action;
        state.state.code[*i] = *parent;
        let i = (0..N)
            .map(|p| PrueferCodeEntry { i: *i, parent: p })
            .map(|a| self.index(&a));
        state.prohibited_actions.extend(i)
    }

    fn action_indices(&self, state: &Self::State) -> impl Iterator<Item = usize> {
        state
            .state
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
            .filter(move |&i| !state.prohibited_actions.contains(&i))
    }

    fn write_vec(&self, state: &Self::State, vec: &mut [f32]) {
        debug_assert!(vec.len() == Self::DIM);
        vec.fill(0.0);
        let (state_vec, action_vec) = vec.split_at_mut(N * (N - 2));
        state
            .state
            .code()
            .iter()
            .enumerate()
            .for_each(|(i, &parent)| {
                let entry = PrueferCodeEntry { i, parent };
                let index = self.index(&entry);
                state_vec[index] = 1.0;
            });
        action_vec.iter_mut().enumerate().for_each(|(i, a)| {
            if state.prohibited_actions.contains(&i) {
                *a = 1.0;
            }
        })
    }

    fn is_terminal(&self, state: &Self::State) -> bool {
        self.action_indices(state).next().is_none()
    }

    fn has_action(&self, state: &Self::State, action: &Self::Action) -> bool {
        let action_index = self.index(action);
        self.action_indices(state).any(|i| i == action_index)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeSet;

    use az_discrete_opt::{space::StateActionSpace, state::prohibit::WithProhibitions};

    use crate::simple_graph::tree::{
        space::modify_each_entry_once::ModifyEachPrueferCodeEntriesExactlyOnce, PrueferCode,
    };

    type SASpace<const N: usize> = ModifyEachPrueferCodeEntriesExactlyOnce<N>;
    // type S<const N: usize> = <ModifyEachPrueferCodeEntriesExactlyOnce<N> as StateActionSpace>::State;
    type A<const N: usize> =
        <ModifyEachPrueferCodeEntriesExactlyOnce<N> as StateActionSpace>::Action;

    #[test]
    fn pruefer_code_indexes_correct_for_4_vertices() {
        let space = ModifyEachPrueferCodeEntriesExactlyOnce::<4>;
        for i in 0..16 {
            let a = space.from_index(i);
            let i2 = space.index(&a);
            assert_eq!(i, i2, "i = {i}, i2 = {i2}, a = {a:?}",);
        }
    }

    #[test]
    fn after_modifying_a_pruefer_code_entry_the_entry_can_no_longer_be_modified() {
        let space = ModifyEachPrueferCodeEntriesExactlyOnce::<4>;
        type A4 = A<4>;
        let mut code = WithProhibitions {
            state: PrueferCode { code: [1, 3, 0, 0] },
            prohibited_actions: BTreeSet::from(
                [
                    // Action { i: 0, parent: 0 },
                    A4 { i: 0, parent: 1 },
                    // Action { i: 0, parent: 2 },
                    // Action { i: 0, parent: 3 },
                    // Action { i: 1, parent: 0 },
                    // Action { i: 1, parent: 1 },
                    // Action { i: 1, parent: 2 },
                    A4 { i: 1, parent: 3 },
                ]
                .map(|a| space.index(&a)),
            ),
        };
        let actions_to_take = [A4 { i: 0, parent: 1 }, A4 { i: 1, parent: 3 }];
        let action_sets: [BTreeSet<A4>; 3] = [
            BTreeSet::from([
                A4 { i: 0, parent: 0 },
                // Action { i: 0, parent: 1 },
                A4 { i: 0, parent: 2 },
                A4 { i: 0, parent: 3 },
                A4 { i: 1, parent: 0 },
                A4 { i: 1, parent: 1 },
                A4 { i: 1, parent: 2 },
                // Action { i: 1, parent: 3 },
            ]),
            BTreeSet::from([
                A4 { i: 1, parent: 0 },
                A4 { i: 1, parent: 1 },
                A4 { i: 1, parent: 2 },
                // Action { i: 1, parent: 3 },
            ]),
            BTreeSet::from([]),
        ];
        // test the action set before taking actions
        let actions = space.action_indices(&code)
            .map(|i| space.from_index(i))
            .collect::<BTreeSet<_>>();
        let (action_set_0, action_sets) = action_sets.split_first().unwrap();
        assert_eq!(actions, *action_set_0);
        for i in 0..2 {
            space.act(&mut code, &actions_to_take[i]);
            let actions = space.action_indices(&code)
                .map(|i| space.from_index(i))
                .collect::<BTreeSet<_>>();
            assert_eq!(actions, action_sets[i]);
        }
    }
}
