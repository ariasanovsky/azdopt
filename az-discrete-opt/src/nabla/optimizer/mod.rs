use super::{space::NablaStateActionSpace, tree::SearchTree, model::NablaModel};

pub struct NablaOptimizer<Space: NablaStateActionSpace, M, P> {
    space: Space,
    roots: Vec<Space::State>,
    states: Vec<Space::State>,
    costs: Vec<Space::Cost>,
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
    ) -> Self
    where
        Space: Sync,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
        P: Send,
    {
        use rayon::{iter::{IntoParallelIterator, ParallelIterator, IntoParallelRefIterator}, slice::{ParallelSlice, ParallelSliceMut}};

        let roots: Vec<_> = (0..batch).into_par_iter().map(|_| init_states()).collect();
        let states = roots.clone();
        let costs = roots.as_slice().par_iter().map(|s| space.cost(s)).collect();
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
                SearchTree::new(&space, s.clone(), c, h_theta)
            })
            .collect();
        Self {
            space,
            roots,
            states,
            costs,
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
    pub fn par_roll_out_episode(&mut self)
    where
        Space: Sync,
        Space::State: Clone + Sync,
        Space::Cost: Send,
    {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        let Self {
            space,
            roots,
            states,
            costs,
            states_host,
            h_theta_host,
            trees,
            model,
        } = self;
        states.clone_from(roots);
        todo!("simulate once");
        todo!("should simulate once update cost, too?");
        (states as &_, costs).into_par_iter().for_each(|(s, c)| {
            *c = space.cost(s);
        });
        todo!()
    }

    pub fn update_model(&mut self, weights: impl Fn(usize) -> f32) -> f32 {
        todo!()
    }
}