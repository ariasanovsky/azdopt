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

    pub fn argmin(&self) -> Option<(&Space::State, &Space::Cost, f32)>
    {
        self.states
            .iter()
            .zip(self.costs.iter())
            .map(|(s, c)| (s, c, self.space.evaluate(c)))
            .min_by(|(_, _, e1), (_, _, e2)| e1.partial_cmp(e2).unwrap())
    }

    #[cfg(feature = "rayon")]
    pub fn par_roll_out_episode(&mut self, max_num_actions: usize)
    where
        Space: Sync,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
        P: Send + Sync + crate::path::ActionPath + crate::path::ActionPathFor<Space>,
    {
        use rayon::{iter::{IntoParallelIterator, ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator}, slice::{ParallelSliceMut, ParallelSlice}};

        use crate::nabla::tree::node::StateNode;
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
        let search_results = (trees, states, paths).into_par_iter().map(|(t, s, p)| {
            t.roll_out_episode(space, s, p)
        }).collect::<Vec<_>>();
        let states = &self.states;
        (states, costs, &search_results).into_par_iter().for_each(|(s, c, (_, kind))| {
            if kind.is_new() {
                *c = space.cost(s);
            }
        });
        let states = &self.states;
        let state_vecs = states_host.par_chunks_exact_mut(Space::STATE_DIM);
        states.par_iter().zip(state_vecs).zip(&search_results).for_each(|((s, s_host), (_, kind))| {
            if kind.is_new() {
                space.write_vec(s, s_host);
            }
        });
        model.write_predictions(states_host, h_theta_host);
        let h_theta_vecs = h_theta_host.par_chunks_exact(Space::ACTION_DIM);
        let new_nodes = (&self.states, &self.costs, &search_results).into_par_iter().zip(h_theta_vecs).map(|((s, c, (_, kind)), h_theta)| {
            if kind.is_new() {
                Some(StateNode::new(space, s, c, h_theta, max_num_actions))
            } else {
                None
            }
        }).collect::<Vec<_>>();
        (search_results, new_nodes).into_par_iter().for_each(|((trans, kind), n)| {
            let c_s_t_theta_star = todo!();
            kind.update_existing_nodes(
                space,
                trans,
                n,
            );
            todo!("update nodes");
        });
        todo!("insert nodes");
        todo!("clear paths")
    }

    pub fn update_model(&mut self, weights: impl Fn(usize) -> f32) -> f32 {
        todo!()
    }
}