pub trait Config {
    type RootData;
    type StateData;
    type Prediction;
    type Path;
    type State;
    type Model;
}

pub trait HasPrediction {
    type P;
}

pub trait HasLog {
    type L;
}

#[macro_export]
macro_rules! impl_config {
    ($config:ty) => {
        impl HasPrediction for $config {
            type P = <$config as Config>::Prediction;
        }
    };
}

#[macro_export]
macro_rules! VisibleRewardTree {
    ($config:ty) => {
        VRewardTree<
            <$config as Config>::State,
            <$config as Config>::Path,
            <$config as Config>::RootData,
            <$config as Config>::StateData,
        >
    };
}