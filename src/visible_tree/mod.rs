use std::collections::BTreeMap;

pub struct VisibleRewardTree<S, P, D0, D> {
    root: S,
    root_data: D0,
    data: BTreeMap<P, D>
}

impl<S, P, D0, D> VisibleRewardTree<S, P, D0, D> {
    pub fn new<Pr: Prediction<S, D0>>(root: S, root_prediction: Pr) -> Self {
        let root_data = root_prediction.data();
        Self {
            root,
            root_data,
            data: BTreeMap::new()
        }
    }
    
}

pub trait Prediction<S, D> {
    fn data(&self) -> D;
}
