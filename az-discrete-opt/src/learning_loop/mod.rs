use crate::{
    az_model::AzModel, int_min_tree::state_data::UpperEstimateData, path::ActionPathFor,
    space::StateActionSpace, state::cost::Cost,
};

use crate::int_min_tree::INTMinTree;

use self::{prediction::PredictionData, state::StateData, tree::TreeData};

pub mod prediction;
pub mod state;
pub mod tree;

pub struct LearningLoop<'a, Space: StateActionSpace, C, P, M: AzModel> {
    pub states: StateData<'a, Space::State, C>,
    pub models: M,
    pub predictions: PredictionData<'a>,
    pub trees: TreeData<P>,
    action: usize,
    gain: usize,
}
impl<'a, Space: StateActionSpace, C, P, M: AzModel> LearningLoop<'a, Space, C, P, M> {
    pub fn new(
        states: StateData<'a, Space::State, C>,
        models: M,
        predictions: PredictionData<'a>,
        trees: TreeData<P>,
        action: usize,
        gain: usize,
    ) -> Self {
        Self {
            states,
            models,
            predictions,
            trees,
            action,
            gain,
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
        space: &Space,
        cost: impl Fn(&Space::State) -> C + Sync,
        upper_estimate: impl Fn(UpperEstimateData) -> f32 + Sync,
    ) where
        Space: StateActionSpace + Sync,
        Space::State: Send + Sync + Clone,
        P: Send + Sync + ActionPathFor<Space> + Ord + Clone,
        C: Cost<f32> + Send + Sync,
    {
        let Self {
            states,
            models,
            predictions,
            trees,
            action,
            gain,
        } = self;
        states.reset_states();
        let ends = trees.par_simulate_once(space, states.get_states_mut(), upper_estimate);
        states.par_write_state_costs(cost);
        states.par_write_state_vecs(space);
        models.write_predictions(states.get_vectors(), predictions);
        ends.par_update_existing_nodes(
            space,
            states.get_costs(),
            states.get_states(),
            predictions,
            *action,
            *gain,
        );
        trees.par_insert_nodes();
    }

    #[cfg(feature = "rayon")]
    pub fn par_update_model(
        &mut self,
        space: &Space,
        logits_mask: Option<&[f32]>,
    ) -> crate::az_model::Loss
    where
        P: Sync,
        Space: StateActionSpace + Sync,
        Space::State: Sync + Clone,
    {
        let Self {
            states,
            models,
            predictions,
            trees,
            action,
            gain,
        } = self;
        use rayon::{
            iter::{IntoParallelIterator, ParallelIterator},
            slice::ParallelSliceMut,
        };

        let (pi_t, g_t) = predictions.get_mut();
        (
            trees.trees(),
            pi_t.par_chunks_exact_mut(*action),
            g_t.par_chunks_exact_mut(*gain),
        )
            .into_par_iter()
            .for_each(|(t, pi_t, g_t)| t.write_observations(pi_t, g_t));
        states.reset_states();
        states.par_write_state_vecs(space);
        models.update_model(states.get_vectors(), logits_mask, predictions)
    }

    #[cfg(feature = "rayon")]
    pub fn par_reset_with_next_root(
        &mut self,
        space: &Space,
        modify_root: impl Fn(usize, &INTMinTree<P>, &mut Space::State) + Sync,
        cost: impl Fn(&Space::State) -> C + Sync,
        add_noise: impl Fn(usize, &mut [f32]) + Sync,
    ) where
        Space: StateActionSpace + Sync,
        Space::State: Send + Sync + Clone,
        C: Cost<f32> + Send + Sync,
        P: Send + Clone,
    {
        let Self {
            states,
            models,
            predictions,
            trees,
            action,
            gain: _,
        } = self;
        use rayon::{
            iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
            slice::ParallelSliceMut,
        };

        (states.get_roots_mut(), trees.trees_mut())
            .into_par_iter()
            .enumerate()
            .for_each(|(i, (s, t))| modify_root(i, t, s));
        states.reset_states();
        states.par_write_state_vecs(space);
        states.par_write_state_costs(cost);
        models.write_predictions(states.get_vectors(), predictions);
        (
            trees.trees_mut(),
            predictions.pi_mut().par_chunks_exact_mut(*action),
            states.get_costs(),
            states.get_states(),
        )
            .into_par_iter()
            .enumerate()
            .for_each(|(i, (t, pi_0_theta, c_0, s_0))| {
                add_noise(i, pi_0_theta);
                t.set_new_root(space, pi_0_theta, c_0.evaluate(), s_0);
            });
    }
}
