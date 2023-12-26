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
    model: M,
}

impl<Space: NablaStateActionSpace, M: NablaModel, P> NablaOptimizer<Space, M, P> {
    #[cfg(feature = "rayon")]
    pub fn par_new(
        space: Space,
        init_states: impl Fn() -> Space::State + Sync + Send,
        mut model: M,
        batch: usize,
        max_num_root_actions: usize,
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
                SearchTree::new(&space, s, c, h_theta, max_num_root_actions)
            })
            .collect();
        Self {
            space,
            roots,
            states,
            costs,
            paths,
            states_host,
            h_theta_host,
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
            states_host: _,
            h_theta_host: _,
            trees: _,
            model: _,
        } = self;
        (states, costs)
            .into_par_iter()
            .map(|(s, c)| (s, c, space.evaluate(c)))
            .min_by(|(_, _, e1), (_, _, e2)| e1.partial_cmp(e2).unwrap())
    }

    #[cfg(feature = "rayon")]
    pub fn par_roll_out_episode(&mut self, max_num_actions: usize)
    where
        Space: Sync,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
        P: Ord + Clone + Send + Sync + crate::path::ActionPath + crate::path::ActionPathFor<Space>,
    {
        use rayon::{iter::{IntoParallelIterator, ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator}, slice::{ParallelSliceMut, ParallelSlice}};

        let Self {
            space,
            roots,
            states,
            costs,
            paths,
            states_host,
            h_theta_host,
            trees,
            model,
        } = self;
        states.clone_from(roots);
        let (transitions, node_kinds): (Vec<_>, Vec<_>) = (trees, states, paths).into_par_iter().map(|(t, s, p)| {
            t.roll_out_episode(space, s, p)
        }).unzip();
        let states = &self.states;
        (states, costs, &node_kinds).into_par_iter().for_each(|(s, c, kind)| {
            if kind.is_new() {
                *c = space.cost(s);
            }
        });
        let states = &self.states;
        let state_vecs = states_host.par_chunks_exact_mut(Space::STATE_DIM);
        states.par_iter().zip(state_vecs).zip(&node_kinds).for_each(|((s, s_host), kind)| {
            if kind.is_new() {
                space.write_vec(s, s_host);
            }
        });
        model.write_predictions(states_host, h_theta_host);
        let h_theta_vecs = h_theta_host.par_chunks_exact(Space::ACTION_DIM);
        let uninserted_new_nodes = (&self.states, &self.costs, &self.paths, node_kinds, transitions).into_par_iter().zip(h_theta_vecs).map(|((s, c, p, kind, trans), h_theta)| {
            kind.update_existing_nodes(
                space, 
                s,
                c,
                h_theta,
                p,
                max_num_actions,
                trans,
            )
        }).collect::<Vec<_>>();
        (&mut self.trees, uninserted_new_nodes, &mut self.paths).into_par_iter().for_each(|(t, n, p)| {
            if let Some(n) = n {
                t.insert_new_node(p.clone(), n);
            }
            p.clear();
        });
        // (&mut self.trees, new_nodes, &mut self.paths, node_kinds).into_par_iter().filter_map(|(t, n, p, k)| {
        //     n.map(|n| (t, n, p, k))
        // }).for_each(|(t, n, p, k)| {
        //     todo!()
        // });
        
        // if let Some(n) = n {
        //     match kind {
        //         NodeKind::NewLevel => todo!(),
        //         NodeKind::New(level) => {
        //             let old_node = level.insert(p.clone(), n);
        //             debug_assert!(old_node.is_none());
        //         },
        //         NodeKind::OldExhausted { c_s_t_theta_star: _ } => unreachable!(),
        //     }
        // }
        // p.clear();
    }

    pub fn update_model<A>(
        &mut self, weights: impl Fn(usize) -> f32,
        cfg: &A,
    ) -> f32 {
        let Self {
            space,
            roots: _,
            states: _,
            costs,
            paths: _,
            states_host,
            h_theta_host,
            trees,
            model,
        } = self;
        todo!("write observations");
        todo!("write weights");
        let observations = todo!();
        model.update_model(states_host, observations)
    }
}