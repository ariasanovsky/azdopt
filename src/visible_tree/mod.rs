use std::collections::BTreeMap;

pub mod config;

use config::*;

pub struct VRewardTree<S, P, D0, D> {
    root: S,
    root_data: D0,
    data: BTreeMap<P, D>
}

impl<S, P, D0, D> VRewardTree<S, P, D0, D> {
    pub fn new<C>(root: S, root_prediction: C::P) -> Self
    where
        C: HasPrediction, 
        C::P: Prediction<D0>,
    {
        let root_data = root_prediction.data();
        Self {
            root,
            root_data,
            data: BTreeMap::new()
        }
    }
    
    pub fn simulate_once<C>(&self, log: &mut C::L)
    where
        C: HasLog,
        C::L: Log,
    {
        todo!()
    }
}

pub trait Prediction<D> {
    fn data(&self) -> D;
}

pub trait Log {

}

pub trait Model<S, P> {
    fn predict(&self, state: &S) -> P;
}
