use super::{space::NablaStateActionSpace, tree::SearchTree, model::NablaModel};

pub struct NablaOptimizer<Space: NablaStateActionSpace, M, P> {
    space: Space,
    roots: Vec<Space::State>,
    states: Vec<Space::State>,
    costs: Vec<Space::Cost>,
    paths: Vec<P>,
    states_host: Vec<f32>,
    h_theta_host: Vec<f32>,
    trees: Vec<SearchTree<P>>,
    action_weights: Vec<f32>,
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
        root_action_pattern: super::tree::node::SamplePattern,
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
        let mut states_host = vec![0.; batch * Space::STATE_DIM];
        (&states, states_host.par_chunks_exact_mut(Space::STATE_DIM)).into_par_iter().for_each(|(s, s_host)| {
            space.write_vec(s, s_host);
        });
        let mut h_theta_host = vec![0.; batch * Space::ACTION_DIM];
        model.write_predictions(&states_host, &mut h_theta_host);
        let trees = 
            (&states, &costs, h_theta_host.par_chunks_exact(Space::ACTION_DIM))
            .into_par_iter()
            .map(|(s, c, h_theta)| {
                SearchTree::new(&space, s, c, h_theta, root_action_pattern.clone())
            })
            .collect();
        let weights = vec![0.; batch * Space::ACTION_DIM];
        Self {
            space,
            roots,
            states,
            costs,
            paths,
            states_host,
            h_theta_host,
            trees,
            action_weights: weights,
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
            states_host: _,
            h_theta_host: _,
            trees: _,
            action_weights: _,
            model: _,
        } = self;
        (states, costs)
            .into_par_iter()
            .map(|(s, c)| (s, c, space.evaluate(c)))
            .min_by(|(_, _, e1), (_, _, e2)| e1.partial_cmp(e2).unwrap())
    }

    #[cfg(feature = "rayon")]
    pub fn par_roll_out_episode(
        &mut self,
        action_pattern: impl Fn(usize) -> super::tree::node::SamplePattern + Sync,
        policy: impl Fn(usize) -> super::tree::node::SearchPolicy + Sync,
    )
    where
        Space: Sync,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
        P: Ord + Clone + Send + Sync + crate::path::ActionPath + crate::path::ActionPathFor<Space>,
    {
        use rayon::{iter::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator}, slice::{ParallelSliceMut, ParallelSlice}};

        use crate::nabla::tree::NodeKind;

        let Self {
            space,
            roots,
            states,
            costs,
            paths,
            states_host,
            h_theta_host,
            trees,
            action_weights: _,
            model,
        } = self;
        // states.clone_from(roots);
        // let (transitions,  node_kinds): (Vec<_>, Vec<_>) = (trees, states, paths).into_par_iter().map(|(t, s, p)| {
        //     t.roll_out_episodes(space, r, s, p, &policy).unzip()
        // }).unzip();
        // let states = &self.states;
        // (states, costs, &node_kinds).into_par_iter().for_each(|(s, c, kind)| {
        //     if kind.as_ref().is_some_and(NodeKind::is_new) {
        //         *c = space.cost(s);
        //     }
        // });
        // let states = &self.states;
        // let state_vecs = states_host.par_chunks_exact_mut(Space::STATE_DIM);
        // (states, state_vecs, &node_kinds).into_par_iter().for_each(|(s, s_host, kind)| {
        //     if kind.as_ref().is_some_and(NodeKind::is_new) {
        //         space.write_vec(s, s_host);
        //     }
        // });
        // model.write_predictions(states_host, h_theta_host);
        // let h_theta_vecs = h_theta_host.par_chunks_exact(Space::ACTION_DIM);
        // let uninserted_new_nodes =
        //     (&self.states, &self.costs, &self.paths, node_kinds, transitions).into_par_iter()
        //     .zip(h_theta_vecs)
        //     .map(|((s, c, p, kind, trans), h_theta)| {
        //     kind.map(|kind| kind.update_existing_nodes(
        //         space, 
        //         s,
        //         c,
        //         h_theta,
        //         p,
        //         action_pattern(p.len()),
        //         trans.unwrap(),
        //     )).flatten()
        // }).collect::<Vec<_>>();
        // (&mut self.trees, uninserted_new_nodes, &mut self.paths).into_par_iter().for_each(|(t, n, p)| {
        //     if let Some(n) = n {
        //         t.insert_new_node(p.clone(), n);
        //     }
        //     p.clear();
        // });
        todo!()
    }

    #[cfg(feature = "rayon")]
    pub fn par_update_model(
        &mut self,
        action_weights: impl Fn(u32) -> f32 + Sync,
        state_weights: impl Fn(&Space::State) -> f32 + Sync,
    ) -> f32
    where
        Space: Sync,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
        P: Sync,
    {
        // use rayon::{iter::{IntoParallelIterator, ParallelIterator, IntoParallelRefIterator}, slice::ParallelSliceMut};

        // // sync `costs` with `roots`
        // (&self.roots, &mut self.costs).into_par_iter().for_each(|(s, c)| {
        //     *c = self.space.cost(s);
        // });

        // let state_weights = &state_weights;
        // let state_weights = self.states.par_iter().map(state_weights).collect::<Vec<_>>();

        // // sync `states_vecs` with `states`
        // let states = &self.states;
        // let state_vecs = self.states_host.par_chunks_exact_mut(Space::STATE_DIM);
        // (states, state_vecs).into_par_iter().for_each(|(s, s_host)| {
        //     self.space.write_vec(s, s_host);
        // });
        // // fill `h_theta_host`
        // self.model.write_predictions(&self.states_host, &mut self.h_theta_host);
        // let h_theta_vecs = self.h_theta_host.par_chunks_exact_mut(Space::ACTION_DIM);
        // // clear weight buffer
        // self.action_weights.fill(0.);
        // let weight_vecs = self.action_weights.par_chunks_exact_mut(Space::ACTION_DIM);
        // // add default weight to valid actions
        // (&self.states, weight_vecs).into_par_iter().for_each(|(s, w)| {
        //     for (a, _) in self.space.action_data(s) {
        //         w[a] = action_weights(0);
        //     }
        // });
        // let weight_vecs = self.action_weights.par_chunks_exact_mut(Space::ACTION_DIM);
        // (&self.trees, &self.roots, &self.costs, weight_vecs, h_theta_vecs).into_par_iter().for_each(|(t, s, c, w, h_theta)| {
        //     let root_node = t.root_node();
        //     let active_actions = root_node.active_actions();
        //     let exhausted_actions = root_node.exhausted_actions();
        //     let action_data = active_actions.chain(exhausted_actions);
        //     for crate::nabla::tree::node::ActionData { a, n_sa, g_theta_star_sa} in action_data {
        //         w[*a] = action_weights(*n_sa);
        //         let r_sa = self.space.reward(s, *a);
        //         h_theta[*a] = self.space.h_sa(c, r_sa, *g_theta_star_sa)
        //     }
        // });
        // self.model.update_model(&self.states_host, &self.h_theta_host, &self.action_weights, &state_weights)
        todo!();
    }

    #[cfg(feature = "rayon")]
    pub fn par_select_next_roots(
        &mut self,
        reset_state: impl Fn(&mut Space::State) + Sync,
        init_state: impl Fn() -> Space::State + Sync + Send,
        root_action_pattern: super::tree::node::SamplePattern,
        label: impl Fn(&Space::State, Option<&P>) -> usize + Sync,
        label_sample: impl Fn(usize) -> super::tree::node::SamplePattern + Sync,
    )
    where
        Space: Sync,
        P: Ord + Send + Sync + crate::path::ActionPathFor<Space>,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
    {
        use rayon::iter::{ParallelIterator, IntoParallelIterator, ParallelExtend, IntoParallelRefMutIterator, IndexedParallelIterator, IntoParallelRefIterator};

        use crate::{path::ActionPath, nabla::tree::node::sample_swap};
        let label = &label;
        let mut next_roots = (&self.trees, &self.roots).into_par_iter().flat_map(|(t, s)| {
            t.par_nodes().map(move |(p, c_star)| (s, p, c_star, label(s, p)))
        }).collect::<Vec<_>>();
        next_roots.sort_unstable_by(|(_, _, c_1, l_1), (_, _, c_2, l_2)| {
            match l_1.cmp(l_2) {
                std::cmp::Ordering::Equal => c_1.partial_cmp(c_2).unwrap(),
                o => o,
            }
        });
        let slices = next_roots.group_by_mut(|(_, _, _, l_1), (_, _, _, l_2)| *l_1 == *l_2).map(|s| {
            (s.first().unwrap().3, s)
        }).collect::<std::collections::BTreeMap<_, _>>();
        for (k, v) in slices.iter() {
            println!("label {k}: {}", v.len());
        }
        let mut next_roots =
            slices
            .into_par_iter()
            .flat_map(|(l, s)| {
                let label_sample = label_sample(l);
                match sample_swap(s, &label_sample) {
                    true => &s[..label_sample.len()],
                    false => s,
                }.par_iter()
            })
            .map(|(s, p, _, _)| {
                let mut s = (*s).clone();
                p.as_deref()
                    .into_iter()
                    .flat_map(ActionPath::actions_taken)
                    .for_each(|a| self.space.act(&mut s, &self.space.action(a)));
                s
            })
            .collect::<Vec<_>>();
        next_roots.truncate(self.roots.len());
        next_roots.par_iter_mut().for_each(|(s)| {
            if self.space.is_terminal(s) {
                reset_state(s);
            }
            while self.space.is_terminal(s) {
                *s = init_state();
            }
        });
        dbg!(next_roots.len());
        let missing = self.roots.len().saturating_sub(next_roots.len());
        next_roots.par_extend((0..missing).into_par_iter().map(|_| {
            init_state()
        }));
        dbg!(next_roots.len());
        // todo!();
        // if sample_swap(next_roots.as_mut_slice(), &root_selection_pattern) {
        //     next_roots.truncate(root_selection_pattern.len());
        // }
        // let mut next_roots = next_roots.into_par_iter().map(|(s, p, _)| {
        //     let mut s = s.clone();
        //     p.into_iter()
        //         .flat_map(ActionPath::actions_taken)
        //         .for_each(|a| self.space.act(&mut s, &self.space.action(a)));
        //     s
        // }).collect::<Vec<_>>();
        self.roots = next_roots;
        (&self.roots, &mut self.costs).into_par_iter().for_each(|(s, c)| {
            *c = self.space.cost(s);
        });
        self.par_reset_trees(root_action_pattern);
        // todo!("anything else?")
    }

    #[cfg(feature = "rayon")]
    fn par_reset_trees(&mut self, root_action_pattern: super::tree::node::SamplePattern)
    where
        Space: Sync,
        Space::State: Sync,
        Space::Cost: Send + Sync,
        P: Send,
    {
        use rayon::{iter::{IntoParallelIterator, ParallelIterator}, slice::ParallelSliceMut};

        (&self.roots, &mut self.costs).into_par_iter().for_each(|(s, c)| {
            *c = self.space.cost(s);
        });
        let state_vecs = self.states_host.par_chunks_exact_mut(Space::STATE_DIM);
        (&self.states, state_vecs).into_par_iter().for_each(|(s, s_host)| {
            self.space.write_vec(s, s_host);
        });
        self.model.write_predictions(&self.states_host, &mut self.h_theta_host);
        let h_theta_vecs = self.h_theta_host.par_chunks_exact_mut(Space::ACTION_DIM);
        (&mut self.trees, &self.roots, &self.costs, h_theta_vecs).into_par_iter().for_each(|(t, s, c, h_theta)| {
            // todo! ?perf
            *t = SearchTree::new(&self.space, s, c, h_theta, root_action_pattern.clone());
        });
        // todo!("anything else?")
    }
}
