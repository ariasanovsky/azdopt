use std::collections::BTreeMap;

use crate::space::StateActionSpace;

use self::state_data::{INTStateData, StateDataKind};

pub mod simulate_once;
pub mod state_data;
pub mod transition;
pub mod update;

#[derive(Debug)]
pub struct INTMinTree<P> {
    pub(crate) root_data: INTStateData,
    pub(crate) data: Vec<BTreeMap<P, StateDataKind>>,
}

impl<P> INTMinTree<P> {
    pub fn len(&self) -> usize {
        self.data.iter().map(|level| level.len()).sum::<usize>()
    }

    pub fn is_empty(&self) -> bool {
        self.data.iter().all(|level| level.is_empty())
    }

    pub fn set_new_root<Space>(
        &mut self,
        space: &Space,
        pi_0_theta: &[f32],
        c_0: f32,
        s_0: &Space::State,
    ) where
        Space: StateActionSpace,
    {
        let Self { root_data, data } = self;
        *root_data = match StateDataKind::new(space, pi_0_theta, c_0, s_0) {
            StateDataKind::Exhausted { c_t: _ } => panic!("root is terminal"),
            StateDataKind::Active { data } => data,
        };
        data.iter_mut().for_each(|level| level.clear());
    }

    pub fn new<Space>(space: &Space, pi_0_theta: &[f32], c_0: f32, s_0: &Space::State) -> Self
    where
        Space: StateActionSpace,
    {
        Self {
            root_data: match StateDataKind::new(space, pi_0_theta, c_0, s_0) {
                StateDataKind::Exhausted { c_t: _ } => panic!("root is terminal"),
                StateDataKind::Active { data } => data,
            },
            data: Vec::new(),
        }
    }

    pub fn print_counts(&self)
    where
        P: core::fmt::Debug,
    {
        let Self { root_data, data } = self;
        println!("root_data.n_s = {}", root_data.n_s);
        for (i, level) in data.iter().enumerate() {
            println!("level {}:", i);
            for (p, data) in level {
                match data {
                    StateDataKind::Exhausted { c_t } => {
                        println!("  {p:?}: exhausted, c_t = {c_t}")
                    }
                    StateDataKind::Active { data } => {
                        println!("  {p:?}: active, n_s = {}", data.n_s)
                    }
                }
            }
        }
    }

    pub fn insert_node_at_next_level(&mut self, level_t: NewTreeLevel<P>)
    where
        P: Ord + Clone,
    {
        let level = BTreeMap::from([(level_t.p_t.clone(), level_t.data_t)]);
        self.data.push(level);
    }

    pub fn write_observations(&self, probs: &mut [f32], values: &mut [f32]) {
        probs.fill(0.0);
        debug_assert_eq!(values.len(), 1);
        let Self { root_data, data: _ } = self;
        root_data.write_observations(probs, values);
    }
}

pub struct NewTreeLevel<P> {
    pub(crate) p_t: P,
    pub(crate) data_t: StateDataKind,
}
