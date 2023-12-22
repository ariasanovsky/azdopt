use super::space::NablaStateActionSpace;

pub struct NablaOptimizer<Space, S, C, M> {
    space: Space,
    roots: Vec<S>,
    states: Vec<S>,
    costs: Vec<C>,
    model: M,
}

impl<Space: NablaStateActionSpace, M> NablaOptimizer<Space, Space::State, Space::Cost, M> {
    pub fn new(
        space: Space,
        init_states: impl Fn() -> Space::State,
        model: M,
        batch: usize,
    ) -> Self
    where
        Space::State: Clone,
    {
        let roots: Vec<_> = (0..batch).map(|_| init_states()).collect();
        let states = roots.clone();
        let costs = roots.iter().map(|s| space.cost(s)).collect();
        Self {
            space,
            roots,
            states,
            costs,
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