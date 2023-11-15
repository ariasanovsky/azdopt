use super::PrueferCode;

#[derive(Debug, PartialEq, Eq)]
pub struct PrueferCodeEntry {
    pub(crate) i: usize,
    pub(crate) parent: usize,
}

impl core::fmt::Display for PrueferCodeEntry {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "({} -> {})", self.i, self.parent)
    }
}

// impl<const N: usize> State for PrueferCode<N> {
//     type Actions = PrueferCodeEntry;

//     fn actions(&self) -> impl Iterator<Item = Self::Actions> {
//         self.code().into_iter().enumerate().map(|(i, parent)| {
//             let before = 0..*parent;
//             let after = *parent+1..N;
//             before.chain(after).map(move |new_parent| PrueferCodeEntry {
//                 i,
//                 parent: new_parent,
//             })
//         }).flatten()
//     }

//     unsafe fn act_unchecked(&mut self, action: &Self::Actions) {
//         let PrueferCodeEntry { i, parent } = action;
//         self.code[*i] = *parent
//     }
// }

// impl<const N: usize> Action<PrueferCode<N>> for PrueferCodeEntry {
//     const DIM: usize = N * (N - 2);

//     fn index(&self) -> usize {
//         let PrueferCodeEntry { i, parent } = self;
//         let i = *i * N + *parent;
//         i
//     }

//     fn from_index(index: usize) -> Self {
//         let i = index / N;
//         let parent = index % N;
//         let i = Self { i, parent };
//         i
//     }

//     fn act(&self, state: &mut PrueferCode<N>) {
//         let PrueferCodeEntry { i, parent } = self;
//         state.code[*i] = *parent
//     }

//     fn actions(state: &PrueferCode<N>) -> impl Iterator<Item = usize> {
//         state.code().iter().enumerate().map(|(i, p)| {
//             let before = 0..*p;
//             let after = *p+1..N;
//             before.chain(after).map(move |new_parent| PrueferCodeEntry {
//                 i,
//                 parent: new_parent,
//             })
//         }).flatten().map(|a| Action::<PrueferCode<N>>::index(&a))
//     }

//     fn is_terminal(state: &PrueferCode<N>) -> bool {
//         Self::actions(state).next().is_none()
//     }

//     // // the entry `a[i] = p` has `i` in `{0, ..., N-3}` and `p` in `{0, ..., N-1}`
//     // fn index(&self) -> usize {
//     //     // dbg!(self);
//     //     let PrueferCodeEntry { i, parent } = self;
//     //     let i = *i * N + *parent;
//     //     // dbg!(i);
//     //     i
//     // }

//     // unsafe fn from_index(index: usize) -> Self {
//     //     // dbg!(index);
//     //     let i = index / N;
//     //     let parent = index % N;
//     //     let i = Self { i, parent };
//     //     // dbg!(&i);
//     //     i
//     // }
// }

// impl<const N: usize> az_discrete_opt::state::StateVec for PrueferCode<N> {
//     const STATE_DIM: usize = N * (N - 2);

//     const ACTION_DIM: usize = N * (N - 2);

//     const WRITE_ACTION_DIMS: bool = false;

//     fn write_vec_state_dims(&self, state_vec: &mut [f32]) {
//         debug_assert_eq!(state_vec.len(), Self::STATE_DIM);
//         state_vec.fill(0.0);
//         for (i, &parent) in self.code().iter().enumerate() {
//             state_vec[i * N + parent] = 1.0;
//         }
//     }

//     fn write_vec_actions_dims(&self, _action_vec: &mut [f32]) {}
// }

#[cfg(test)]
mod tests {
    use crate::simple_graph::tree::PrueferCode;

    use super::PrueferCodeEntry;

    #[test]
    fn test_pruefer_codes_for_trees_on_3_vertices_have_correct_actions() {
        todo!();
        // let codes = [
        //     [0, 0, 0], // extra 2 entries should be irrelevant
        //     [1, 1, 1],
        //     [2, 2, 2],
        // ].map(|code| PrueferCode::<3> { code });
        // let correct_actions = [
        //     [
        //         // (0, 0),
        //         (0, 1),
        //         (0, 2),
        //     ],
        //     [
        //         (0, 0),
        //         // (0, 1),
        //         (0, 2),
        //     ],
        //     [
        //         (0, 0),
        //         (0, 1),
        //         // (0, 2),
        //     ],
        // ].map(|actions| {
        //     actions.map(|(i, parent)| PrueferCodeEntry { i, parent })
        // });
        // for (code, correct_actions) in codes.iter().zip(correct_actions.iter()) {
        //     let actions: Vec<PrueferCodeEntry> = code.actions::<PrueferCodeEntry>().into_iter().map(|i| {
        //         let i = PrueferCode::<3>::from_index(i);
        //         i
        //     }).collect();
        //     assert_eq!(&actions, correct_actions);
        // }
    }
}