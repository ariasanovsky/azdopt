use core::mem::MaybeUninit;

use rayon::prelude::*;

use crate::{int_min_tree::INTMinTree, space::StateActionSpace, path::ActionPathFor};

pub fn par_set_next_roots<const B: usize, Space, P>(
    s_0: &mut [Space::State; B],
    tree: &[INTMinTree<P>; B],
)
where
    Space: StateActionSpace,
    Space::State: Send + Sync + Clone,
    P: Sync + ActionPathFor<Space>,
{
    let mut candidates: [_; B] = core::array::from_fn(|i| Vec::with_capacity(tree[i].len() - 1));
    (
        s_0 as &[Space::State; B],
        tree,
        &mut candidates,
    )
        .into_par_iter()
        .for_each(|(s, t, v)| {
            v.extend(
                t.data
                .iter()
                .flat_map(|level| level.iter())
                .map(|(p, data)| (s, p, data)),
            );
        });
    let mut candidates = candidates.into_par_iter().flatten().collect::<Vec<_>>();
    
    candidates.par_sort_unstable_by(|a, b| {
        let a_cost = a.2.cost();
        let b_cost = b.2.cost();
        a_cost.partial_cmp(&b_cost).unwrap()
    });
    println!("candidates: {:?}", candidates.len());
    let chunk_size = candidates.len().div_ceil(B);
    let chunks = candidates.par_chunks(chunk_size);
    let mut roots = MaybeUninit::uninit_array();
    let count = (&mut roots, chunks).into_par_iter().map(|(root, chunk)| {
        let (s, p, _) = *chunk.first().unwrap();
        let mut s = s.clone();
        Space::follow(&mut s, p.actions_taken().map(|a| Space::from_index(*a)));
        root.write(s);
    }).count();
    assert_eq!(count, B);
    let roots = unsafe { MaybeUninit::array_assume_init(roots) };
    *s_0 = roots;
}
