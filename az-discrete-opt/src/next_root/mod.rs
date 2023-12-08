use crate::int_min_tree::{state_data::StateDataKind, INTMinTree};

impl<P> INTMinTree<P> {
    pub fn unstable_sorted_nodes(&self) -> Vec<(&P, &StateDataKind)> {
        let mut nodes = self
            .data
            .iter()
            .flat_map(|level| level.iter())
            .collect::<Vec<_>>();
        nodes.sort_unstable_by(|a, b| {
            let a_cost = a.1.cost();
            let b_cost = b.1.cost();
            a_cost.partial_cmp(&b_cost).unwrap()
        });
        nodes
    }
}

// pub fn par_set_next_roots<const B: usize, Space, P>(
//     s_0: &mut [Space::State; B],
//     tree: &[INTMinTree<P>; B],
//     greed_policy: impl(Fn(usize, (usize, f32), (usize, f32)) -> bool) + Sync + Send,
// )
// where
//     Space: StateActionSpace,
//     Space::State: Send + Sync + Clone,
//     P: Sync + ActionPathFor<Space>,
// {

//     tree[0].data.drain(range)

//     let mut candidates: [_; B] = core::array::from_fn(|i| Vec::with_capacity(tree[i].len() - 1));
//     let mut cost_sum: [f32; B] = [0.0; B];
//     (
//         tree,
//         &mut candidates,
//         &mut cost_sum,
//     )
//         .into_par_iter()
//         .for_each(|(t, v, c_sum)| {
//             v.extend(
//                 t.data
//                 .iter()
//                 .flat_map(|level| level.iter())
//             );
//             v.sort_unstable_by(|a, b| {
//                 let a_cost = a.1.cost();
//                 let b_cost = b.1.cost();
//                 a_cost.partial_cmp(&b_cost).unwrap()
//             });
//             *c_sum = v.iter().map(|(_, p)| p.cost()).sum();
//         });
//     (s_0, candidates, cost_sum).into_par_iter().enumerate().for_each(|(j, (s, c, c_sum))| {
//         let mut running_sum = 0.0;
//         let len = c.len();
//         let next_path = c.iter().enumerate().find(|(i, (_, k))| {
//             let choose = greed_policy(j, (*i, running_sum), (len, c_sum));
//             running_sum += k.cost();
//             choose
//         }).unwrap().1.0;
//         Space::follow(s, next_path.actions_taken().map(|a| Space::from_index(*a)));
//     })
// }
