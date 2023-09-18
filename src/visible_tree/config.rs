pub trait Config {
    type RootData;
    type StateData;
    type Prediction;
    type Path;
    type State;
    type Model;
    type Reward;
}

pub trait HasPrediction {
    type P;
}

pub trait HasLog {
    type L;
}

pub trait HasReward {
    type R;
}

// #[macro_export]
// macro_rules! impl_config {
//     ($config:ty) => {
//         impl HasPrediction for $config {
//             type P = <$config as Config>::Prediction;
//         }
//         impl HasReward for $config {
//             type R = <$config as Config>::Reward;
//         }
//     };
// }

impl<C: Config> HasPrediction for C {
    type P = C::Prediction;
}

impl<C: Config> HasReward for C {
    type R = C::Reward;
}


#[macro_export]
macro_rules! VisibleRewardTree {
    ($config:ty) => {
        VRewardTree<
            <$config as $crate::visible_tree::config::Config>::State,
            <$config as $crate::visible_tree::config::Config>::Path,
            <$config as $crate::visible_tree::config::Config>::RootData,
            <$config as $crate::visible_tree::config::Config>::StateData,
        >
    };
}