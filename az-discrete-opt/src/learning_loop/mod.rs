use crate::{
    az_model::AzModel, int_min_tree::state_data::UpperEstimateData, path::ActionPathFor,
    space::StateActionSpace, state::cost::Cost,
};

use self::{prediction::PredictionData, state::StateData, tree::TreeData};

pub mod prediction;
pub mod state;
pub mod tree;

pub fn par_roll_out_episode<
    const BATCH: usize,
    const STATE: usize,
    const ACTION: usize,
    const GAIN: usize,
    Space,
    C,
    P,
>(
    states: &mut StateData<BATCH, STATE, Space::State, C>,
    models: &mut impl AzModel<BATCH, STATE, ACTION, GAIN>,
    predictions: &mut PredictionData<BATCH, ACTION, GAIN>,
    trees: &mut TreeData<BATCH, P>,
    cost: impl Fn(&Space::State) -> C + Sync,
    upper_estimate: impl Fn(UpperEstimateData) -> f32 + Sync,
) where
    Space: StateActionSpace,
    Space::State: Send + Sync + Clone,
    P: Send + Sync + ActionPathFor<Space> + Ord + Clone,
    C: Cost<f32> + Send + Sync,
{
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
    trees.insert_nodes();
}
