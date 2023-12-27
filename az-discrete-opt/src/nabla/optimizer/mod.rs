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
    weights: Vec<f32>,
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
            weights,
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
            weights: _,
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
        use rayon::{iter::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator}, slice::{ParallelSliceMut, ParallelSlice}};

        let Self {
            space,
            roots,
            states,
            costs,
            paths,
            states_host,
            h_theta_host,
            trees,
            weights: _,
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
        (states, state_vecs, &node_kinds).into_par_iter().for_each(|(s, s_host, kind)| {
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
    }

    #[cfg(feature = "rayon")]
    pub fn par_update_model(&mut self, weight_fn: impl Fn(u32) -> f32 + Sync) -> f32
    where
        Space: Sync,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
        P: Sync,
    {
        use rayon::{iter::{IntoParallelIterator, ParallelIterator}, slice::ParallelSliceMut};

        // sync `costs` with `roots`
        (&self.roots, &mut self.costs).into_par_iter().for_each(|(s, c)| {
            *c = self.space.cost(s);
        });
        // sync `states_vecs` with `states`
        let states = &self.states;
        let state_vecs = self.states_host.par_chunks_exact_mut(Space::STATE_DIM);
        (states, state_vecs).into_par_iter().for_each(|(s, s_host)| {
            self.space.write_vec(s, s_host);
        });
        // fill `h_theta_host`
        self.model.write_predictions(&self.states_host, &mut self.h_theta_host);
        let h_theta_vecs = self.h_theta_host.par_chunks_exact_mut(Space::ACTION_DIM);
        // clear weight buffer
        self.weights.fill(0.);
        let weight_vecs = self.weights.par_chunks_exact_mut(Space::ACTION_DIM);
        // add default weight to valid actions
        (&self.states, weight_vecs).into_par_iter().for_each(|(s, w)| {
            for (a, _) in self.space.action_data(s) {
                w[a] = weight_fn(0);
            }
        });
        let weight_vecs = self.weights.par_chunks_exact_mut(Space::ACTION_DIM);
        (&self.trees, &self.roots, &self.costs, weight_vecs, h_theta_vecs).into_par_iter().for_each(|(t, s, c, w, h_theta)| {
            let root_node = t.root_node();
            let active_actions = root_node.active_actions();
            let exhausted_actions = root_node.exhausted_actions();
            let action_data = active_actions.chain(exhausted_actions);
            for crate::nabla::tree::node::ActionData { a, n_sa, g_theta_star_sa} in action_data {
                w[*a] = weight_fn(*n_sa);
                let r_sa = self.space.reward(s, *a);
                h_theta[*a] = self.space.h_sa(c, r_sa, *g_theta_star_sa)
            }
        });
        self.model.update_model(&self.states_host, &self.h_theta_host, &self.weights)
    }
}