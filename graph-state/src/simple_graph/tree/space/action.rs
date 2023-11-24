#[derive(Debug, PartialEq, Eq, PartialOrd, Ord)]
pub struct PrueferCodeEntry {
    pub(crate) i: usize,
    pub(crate) parent: usize,
}

impl PrueferCodeEntry {
    pub fn action_index<const N: usize>(&self) -> usize {
        let PrueferCodeEntry { i, parent } = self;
        *i * N + *parent
    }

    pub fn from_action_index<const N: usize>(index: usize) -> Self {
        debug_assert!(index < N * (N - 2));
        let i = index / N;
        let parent = index % N;
        PrueferCodeEntry { i, parent }
    }

    pub fn indices_for_the_same_entry<const N: usize>(&self) -> [usize; N] {
        let PrueferCodeEntry { i, parent: _ } = self;
        core::array::from_fn(|j| N * i + j)
    }
}

impl core::fmt::Display for PrueferCodeEntry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let PrueferCodeEntry { i, parent } = self;
        write!(f, "c[{i}] = {parent}")
    }
}

#[cfg(test)]
mod tests {
    use az_discrete_opt::space::StateActionSpace;

    use crate::simple_graph::tree::{
        space::modify_any_entry::ModifyAnyPrueferCodeEntry, PrueferCode,
    };

    use super::PrueferCodeEntry;

    #[test]
    fn test_pruefer_codes_for_trees_on_3_vertices_have_correct_actions() {
        let codes = [
            [0, 0, 0], // extra 2 entries should be irrelevant
            [1, 1, 1],
            [2, 2, 2],
        ]
        .map(|code| PrueferCode::<3> { code });
        let correct_actions = [
            [
                // (0, 0),
                (0, 1),
                (0, 2),
            ],
            [
                (0, 0),
                // (0, 1),
                (0, 2),
            ],
            [
                (0, 0),
                (0, 1),
                // (0, 2),
            ],
        ]
        .map(|actions| actions.map(|(i, parent)| PrueferCodeEntry { i, parent }));
        type Space = ModifyAnyPrueferCodeEntry<3>;
        // type Action = PrueferCodeEntry;
        for (code, correct_actions) in codes.iter().zip(correct_actions.iter()) {
            let actions: Vec<PrueferCodeEntry> = Space::action_indices(&code)
                .map(|i| Space::from_index(i))
                .collect();
            assert_eq!(actions, correct_actions);
            // todo!();
        }
    }
}
