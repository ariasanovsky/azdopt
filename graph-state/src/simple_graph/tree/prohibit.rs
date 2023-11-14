// use az_discrete_opt::{tree_node::prohibitions::ProhibitActionsFor, path::{ActionMultiset, ActionSet}};

// use super::{state::PrueferCodeEntry, PrueferCode};

// /// If we prohibit modifying the same entry in the Pruefer code twice, then we can treat paths as multisets of actions.
// impl<const N: usize> ProhibitActionsFor<PrueferCodeEntry, ActionMultiset> for PrueferCode<N> {
//     fn update_prohibited_actions(
//         &self,
//         prohibited_actions: &mut std::collections::BTreeSet<usize>,
//         action: &PrueferCodeEntry,
//     ) {
//         dbg!(action);
//         let PrueferCodeEntry { i, parent: _ } = action;
//         prohibited_actions.extend(
//             (0..N)
//             .map(|p| *i * N + p)
//         );
//     }
// }

// /// In fact, this also allows us to treat paths as sets of actions.
// impl<const N: usize> ProhibitActionsFor<PrueferCodeEntry, ActionSet> for PrueferCode<N> {
//     fn update_prohibited_actions(
//         &self,
//         prohibited_actions: &mut std::collections::BTreeSet<usize>,
//         action: &PrueferCodeEntry,
//     ) {
//         <Self as ProhibitActionsFor<PrueferCodeEntry, ActionMultiset>>::update_prohibited_actions(
//             self,
//             prohibited_actions,
//             action,
//         );
//     }
// }