use core::num::NonZeroUsize;

use super::{space::NablaStateActionSpace, tree::{SearchTree, Transition2}, model::NablaModel};

pub struct NablaOptimizer<Space: NablaStateActionSpace, M, P> {
    space: Space,
    roots: Vec<Space::State>,
    states: Vec<Space::State>,
    costs: Vec<Space::Cost>,
    paths: Vec<P>,
    transitions: Vec<Vec<Transition2>>,
    last_positions: Vec<Option<NonZeroUsize>>,
    state_vecs: Vec<f32>,
    h_theta_host: Vec<f32>,
    action_weights: Vec<f32>,
    trees: Vec<SearchTree<P>>,
    model: M,
}

impl<Space: NablaStateActionSpace, M: NablaModel, P> NablaOptimizer<Space, M, P> {
    pub fn get_model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    pub fn get_trees(&self) -> &[SearchTree<P>] {
        &self.trees
    }
    
    #[cfg(feature = "rayon")]
    pub fn par_new(
        space: Space,
        init_states: impl Fn() -> Space::State + Sync + Send,
        mut model: M,
        batch: usize,
    ) -> Self
    where
        Space: Sync,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
        P: Send + crate::path::ActionPath,
    {
        use rayon::{iter::{IntoParallelIterator, ParallelIterator, IntoParallelRefIterator}, slice::{ParallelSlice, ParallelSliceMut}};

        let roots: Vec<_> = (0..batch).into_par_iter().map(|_| init_states()).collect();
        let states = roots.clone();
        let costs = roots.as_slice().par_iter().map(|s| space.cost(s)).collect();
        let paths = (0..batch).into_par_iter().map(|_| P::new()).collect();
        let mut state_vecs = vec![0.; batch * Space::STATE_DIM];
        (&states, state_vecs.par_chunks_exact_mut(Space::STATE_DIM)).into_par_iter().for_each(|(s, s_host)| {
            space.write_vec(s, s_host);
        });
        let mut h_theta_host = vec![0.; batch * Space::ACTION_DIM];
        model.write_predictions(&state_vecs, &mut h_theta_host);
        // h_theta_host.par_iter_mut().for_each(|x| *x = x.sqrt());
        let trees = 
            (&states, &costs, h_theta_host.par_chunks_exact(Space::ACTION_DIM))
            .into_par_iter()
            .map(|(s, c, h_theta)| {
                SearchTree::new(&space, s, c, h_theta)
            })
            .collect();
        let transitions = (0..batch).into_par_iter().map(|_| Vec::new()).collect();
        let last_positions = vec![None; batch];
        let action_weights = vec![0.; batch * Space::ACTION_DIM];
        // dbg!(action_weights.len());
        Self {
            space,
            roots,
            states,
            costs,
            paths,
            transitions,
            last_positions,
            state_vecs,
            h_theta_host,
            action_weights,
            trees,
            model,
        }
    }

    #[cfg(feature = "rayon")]
    pub fn par_argmin(&self) -> Option<(&Space::State, &Space::Cost, f32)>
    where
        Space: Sync,
        Space::State: Send + Sync,
        Space::Cost: Send + Sync,
    {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let Self {
            space,
            roots: _,
            states,
            costs,
            paths: _,
            transitions: _,
            last_positions: _,
            state_vecs: _,
            h_theta_host: _,
            action_weights: _,
            trees: _,
            model: _,
        } = self;
        (states, costs)
            .into_par_iter()
            .map(|(s, c)| (s, c, space.evaluate(c)))
            .min_by(|(_, _, e1), (_, _, e2)| e1.partial_cmp(e2).unwrap())
    }

    #[cfg(feature = "rayon")]
    pub fn par_roll_out_episodes(
        &mut self,
        // action_pattern: impl Fn(usize) -> super::tree::node::SamplePattern + Sync,
        // policy: impl Fn(usize) -> super::tree::node::SearchPolicy + Sync,
    )
    where
        Space: Sync,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
        P: Ord + Clone + Send + Sync + crate::path::ActionPath + crate::path::ActionPathFor<Space>,
    {
        use rayon::{iter::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator}, slice::{ParallelSliceMut, ParallelSlice}};

        use crate::nabla::tree::node::StateNode2;

        let trees = &mut self.trees;
        let roots = &self.roots;
        let states = &mut self.states;
        let paths = &mut self.paths;
        let transitions = &mut self.transitions;
        let costs = &mut self.costs;
        let last_positions = &mut self.last_positions;
        let state_vecs = self.state_vecs.par_chunks_exact_mut(Space::STATE_DIM);
        debug_assert!(
            [
                roots.len(),
                states.len(),
                paths.len(),
                transitions.len(),
                costs.len(),
                last_positions.len(),
                state_vecs.len(),
            ]
            .into_iter()
            .all(|l| l == trees.len()),
        );
        (
            trees,
            roots,
            states,
            paths,
            transitions,
            costs,
            last_positions,
            state_vecs,
        ).into_par_iter().for_each(|(
            t,
            r,
            s,
            p,
            trans,
            c,
            pos,
            v,
        )| {
            t.roll_out_episodes(
                &self.space,
                r,
                s,
                c,
                p,
                trans,
                pos,
            );    
            if !trans.is_empty() {
                self.space.write_vec(s, v);
            }
        });
        self.model.write_predictions(&self.state_vecs, &mut self.h_theta_host);
        (
            &mut self.trees,
            &self.states,
            &self.costs,
            &self.transitions,
            self.h_theta_host.par_chunks_exact(Space::ACTION_DIM),
        ).into_par_iter().for_each(|(
            t,
            s,
            c,
            trans,
            h,
        )| {
            if !trans.is_empty() {
                let n = StateNode2::new(&self.space, s, c, h);
                t.push_node(n);
            }
        });
        // todo!();
        // (&mut self.trees, uninserted_new_nodes, &mut self.paths).into_par_iter().for_each(|(t, n, p)| {
        //     if let Some(n) = n {
        //         t.insert_new_node(p.clone(), n);
        //     }
        //     p.clear();
        // });
    }

    #[cfg(feature = "rayon")]
    pub fn par_update_model(
        &mut self,
        // action_weights: impl Fn(u32) -> f32 + Sync,
        // state_weights: impl Fn(&Space::State) -> f32 + Sync,
    ) -> f32
    where
        Space: Sync,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
        P: Sync,
    {
        use rayon::{iter::{IntoParallelIterator, ParallelIterator}, slice::ParallelSliceMut};

        // sync `costs` with `roots`
        let state_vecs = self.state_vecs.par_chunks_exact_mut(Space::STATE_DIM);
        (
            &self.roots,
            state_vecs,
        ).into_par_iter().for_each(|(s, v)| {
            self.space.write_vec(s, v);
        });

        // fill `h_theta_host`
        self.h_theta_host.fill(0.0);
        self.action_weights.fill(0.0);
        let h_theta_vecs = self.h_theta_host.par_chunks_exact_mut(Space::ACTION_DIM);
        let weight_vecs = self.action_weights.par_chunks_exact_mut(Space::ACTION_DIM);
        (&self.trees, h_theta_vecs, weight_vecs)
            .into_par_iter()
            .for_each(|(t, h_theta, weights)|
                t.write_observations(&self.space, h_theta, weights)
            );
        self.model.update_model(&self.state_vecs, &self.h_theta_host, &self.action_weights)
    }

    #[cfg(feature = "rayon")]
    pub fn reset_trees(
        &mut self,
        modify_root: impl Fn(&Space, &mut Space::State, Vec<(Option<&P>, f32, f32)>) + Sync,
    )
    where
        Space: Sync,
        P: Ord + Send + Sync + crate::path::ActionPathFor<Space>,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
    {
        use rayon::{iter::{ParallelIterator, IntoParallelIterator, ParallelExtend, IntoParallelRefMutIterator, IntoParallelRefIterator}, slice::{ParallelSliceMut, ParallelSlice}};

        let Self {
            space: _,
            roots,
            states,
            costs,
            paths,
            transitions,
            last_positions,
            state_vecs,
            h_theta_host,
            action_weights: _,
            trees: _,
            model,
        } = self;
        last_positions.fill(None);
        state_vecs.fill(0.);
        let space = &self.space;
        let trees = &self.trees;
        let state_vecs = self.state_vecs.par_chunks_exact_mut(Space::STATE_DIM);
        let modify_root = &modify_root;
        (
            trees,
            roots,
            states,
            costs,
            paths,
            transitions,
            state_vecs,
        ).into_par_iter().for_each(|(
            t,
            r,
            s,
            c,
            p,
            trans,
            v,
        )| {
            let n = t.node_data();
            modify_root(space, r, n);
            s.clone_from(r);
            *c = space.cost(r);
            self.space.write_vec(s, v);
            p.clear();
            trans.clear();
        });
        h_theta_host.fill(0.);
        model.write_predictions(&self.state_vecs, h_theta_host);
        let action_vecs = h_theta_host.par_chunks_exact(Space::ACTION_DIM);
        (
            &mut self.trees,
            &self.roots,
            &self.costs,
            action_vecs,
        ).into_par_iter().for_each(|(
            t,
            r,
            c,
            h,
        )| {
            *t = SearchTree::new(&self.space, r, c, h);
        });
        // let mut next_roots = (&self.trees, &self.roots).into_par_iter().flat_map(|(t, s)| {
        //     t.par_nodes().map(move |(p, c_star)| (s, p, c_star, label(s, p)))
        // }).collect::<Vec<_>>();
        // next_roots.sort_unstable_by(|(_, _, c_1, l_1), (_, _, c_2, l_2)| {
        //     match l_1.cmp(l_2) {
        //         std::cmp::Ordering::Equal => c_1.partial_cmp(c_2).unwrap(),
        //         o => o,
        //     }
        // });
        // let slices = next_roots.group_by_mut(|(_, _, _, l_1), (_, _, _, l_2)| *l_1 == *l_2).map(|s| {
        //     (s.first().unwrap().3, s)
        // }).collect::<std::collections::BTreeMap<_, _>>();
        // // for (k, v) in slices.iter() {
        // //     println!("label {k}: {}", v.len());
        // // }
        // let mut next_roots =
        //     slices
        //     .into_par_iter()
        //     .flat_map(|(l, s)| {
        //         let label_sample = label_sample(l);
        //         match sample_swap(s, &label_sample) {
        //             true => &s[..label_sample.len()],
        //             false => s,
        //         }.par_iter()
        //     })
        //     .map(|(s, p, _, _)| {
        //         let mut s = (*s).clone();
        //         p.as_deref()
        //             .into_iter()
        //             .flat_map(ActionPath::actions_taken)
        //             .for_each(|a| self.space.act(&mut s, &self.space.action(a)));
        //         s
        //     })
        //     .collect::<Vec<_>>();
        // next_roots.truncate(self.roots.len());
        // dbg!(next_roots.len());
        // next_roots.par_iter_mut().for_each(|s| {
        //     if self.space.is_terminal(s) {
        //         reset_state(s);
        //     }
        //     while self.space.is_terminal(s) {
        //         *s = init_state();
        //     }
        // });
        // dbg!(next_roots.len());
        // let missing = self.roots.len().saturating_sub(next_roots.len());
        // next_roots.par_extend((0..missing).into_par_iter().map(|_| {
        //     init_state()
        // }));
        // dbg!(next_roots.len());
        // // todo!();
        // // if sample_swap(next_roots.as_mut_slice(), &root_selection_pattern) {
        // //     next_roots.truncate(root_selection_pattern.len());
        // // }
        // // let mut next_roots = next_roots.into_par_iter().map(|(s, p, _)| {
        // //     let mut s = s.clone();
        // //     p.into_iter()
        // //         .flat_map(ActionPath::actions_taken)
        // //         .for_each(|a| self.space.act(&mut s, &self.space.action(a)));
        // //     s
        // // }).collect::<Vec<_>>();
        // self.roots = next_roots;
        // (&self.roots, &mut self.costs).into_par_iter().for_each(|(s, c)| {
        //     *c = self.space.cost(s);
        // });
        // self.par_reset_trees();
        // todo!("anything else?")
    }

    #[cfg(feature = "rayon")]
    fn par_reset_trees(&mut self)
    where
        Space: Sync,
        Space::State: Sync,
        Space::Cost: Send + Sync,
        P: Send + crate::path::ActionPath,
    {
        use rayon::{iter::{IntoParallelIterator, ParallelIterator}, slice::ParallelSliceMut};

        (&self.roots, &mut self.costs).into_par_iter().for_each(|(s, c)| {
            *c = self.space.cost(s);
        });
        let state_vecs = self.state_vecs.par_chunks_exact_mut(Space::STATE_DIM);
        (&self.states, state_vecs).into_par_iter().for_each(|(s, s_host)| {
            self.space.write_vec(s, s_host);
        });
        self.model.write_predictions(&self.state_vecs, &mut self.h_theta_host);
        let h_theta_vecs = self.h_theta_host.par_chunks_exact_mut(Space::ACTION_DIM);
        (&mut self.trees, &self.roots, &self.costs, h_theta_vecs).into_par_iter().for_each(|(t, s, c, h_theta)| {
            // todo! ?perf
            // dbg!();
            *t = SearchTree::new(&self.space, s, c, h_theta);
        });
        self.last_positions.fill(None);
        (&mut self.paths, &mut self.transitions)
            .into_par_iter()
            .for_each(|(p, t)| {
                p.clear();
                t.clear();
            });
        // todo!("anything else?")
    }
}
