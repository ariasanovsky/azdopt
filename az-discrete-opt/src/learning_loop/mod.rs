use crate::{
    az_model::AzModel, int_min_tree::state_data::UpperEstimateData, path::ActionPathFor,
    space::StateActionSpace, state::cost::Cost,
};

use crate::int_min_tree::INTMinTree;

use self::{prediction::PredictionData, state::StateData, tree::TreeData};

pub mod prediction;
pub mod state;
pub mod tree;

pub struct LearningLoop<
    const BATCH: usize,
    const STATE: usize,
    const ACTION: usize,
    const GAIN: usize,
    Space: StateActionSpace,
    C,
    P,
    M: AzModel<BATCH, STATE, ACTION, GAIN>,
> {
    pub states: StateData<BATCH, STATE, Space::State, C>,
    pub models: M,
    pub predictions: PredictionData<BATCH, ACTION, GAIN>,
    pub trees: TreeData<BATCH, P>,
}
impl<
        const BATCH: usize,
        const STATE: usize,
        const ACTION: usize,
        const GAIN: usize,
        Space: StateActionSpace,
        C,
        P,
        M: AzModel<BATCH, STATE, ACTION, GAIN>,
    > LearningLoop<BATCH, STATE, ACTION, GAIN, Space, C, P, M>
{
    pub fn new(
        states: StateData<BATCH, STATE, Space::State, C>,
        models: M,
        predictions: PredictionData<BATCH, ACTION, GAIN>,
        trees: TreeData<BATCH, P>,
    ) -> Self {
        Self {
            states,
            models,
            predictions,
            trees,
        }
    }

    // pub fn get_states(&self) -> &[Space::State; BATCH] {
    //     self.states.get_states()
    // }

    // pub fn get_costs(&self) -> &[C; BATCH] {
    //     self.states.get_costs()
    // }

    #[cfg(feature = "rayon")]
    pub fn par_argmin(&self) -> Option<(&Space::State, &C)>
    where
        Space: StateActionSpace,
        Space::State: Send + Sync + Clone,
        C: Cost<f32> + Send + Sync,
    {
        use rayon::iter::{IntoParallelIterator, ParallelIterator};
        (self.states.get_states(), self.states.get_costs())
            .into_par_iter()
            .min_by(|(_, a), (_, b)| a.evaluate().partial_cmp(&b.evaluate()).unwrap())
    }

    #[cfg(feature = "rayon")]
    pub fn par_roll_out_episode(
        &mut self,
        cost: impl Fn(&Space::State) -> C + Sync,
        upper_estimate: impl Fn(UpperEstimateData) -> f32 + Sync,
    ) where
        Space: StateActionSpace,
        Space::State: Send + Sync + Clone,
        P: Send + Sync + ActionPathFor<Space> + Ord + Clone,
        C: Cost<f32> + Send + Sync,
    {
        let Self {
            states,
            models,
            predictions,
            trees,
        } = self;
        states.reset_states();
        let ends = trees.par_simulate_once::<Space>(states.get_states_mut(), upper_estimate);
        states.par_write_state_costs(cost);
        states.par_write_state_vecs::<Space>();
        models.write_predictions(states.get_vectors(), predictions);
        ends.par_update_existing_nodes::<Space, ACTION, GAIN>(
            states.get_costs(),
            states.get_states(),
            predictions,
        );
        trees.par_insert_nodes();
    }

    #[cfg(feature = "rayon")]
    pub fn par_update_model(&mut self) -> crate::az_model::Loss
    where
        P: Sync,
        Space: StateActionSpace,
        Space::State: Sync + Clone,
    {
        let Self {
            states,
            models,
            predictions,
            trees,
        } = self;
        use rayon::iter::{IntoParallelIterator, ParallelIterator};

        let (pi_t, g_t) = predictions.get_mut();
        (trees.trees(), pi_t, g_t)
            .into_par_iter()
            .for_each(|(t, pi_t, g_t)| t.write_observations(pi_t, g_t));
        states.reset_states();
        states.par_write_state_vecs::<Space>();
        models.update_model(states.get_vectors(), predictions)
    }

    #[cfg(feature = "rayon")]
    pub fn par_reset_with_next_root(
        &mut self,
        modify_root: impl Fn(usize, &INTMinTree<P>, &mut Space::State) + Sync,
        cost: impl Fn(&Space::State) -> C + Sync,
        add_noise: impl Fn(usize, &mut [f32]) + Sync,
    ) where
        Space: StateActionSpace,
        Space::State: Send + Sync + Clone,
        C: Cost<f32> + Send + Sync,
        P: Send + Clone,
    {
        let Self {
            states,
            models,
            predictions,
            trees,
        } = self;
        use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

        (states.get_roots_mut(), trees.trees_mut())
            .into_par_iter()
            .enumerate()
            .for_each(|(i, (s, t))| modify_root(i, t, s));
        states.reset_states();
        states.par_write_state_vecs::<Space>();
        states.par_write_state_costs(cost);
        models.write_predictions(states.get_vectors(), predictions);
        (
            trees.trees_mut(),
            predictions.pi_mut(),
            states.get_costs(),
            states.get_states(),
        )
            .into_par_iter()
            .enumerate()
            .for_each(|(i, (t, pi_0_theta, c_0, s_0))| {
                add_noise(i, pi_0_theta);
                t.set_new_root::<Space>(pi_0_theta, c_0.evaluate(), s_0);
            });
    }
}
