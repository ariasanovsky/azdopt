use super::HasExpectedFutureGain;

pub trait Config {
    type RootData;
    type StateData;
    type Prediction;
    type Path;
    type State;
    type Model;
    type Reward;
    type ExpectedFutureGain;
    /* type VRewardTree = VisibleRewardTree<
        Self::State,
        Self::Path,
        Self::RootData,
        Self::StateData,
    >; 
    = note: see issue #29661 <https://github.com/rust-lang/rust/issues/29661> for more information
    = help: add `#![feature(associated_type_defaults)]` to the crate attributes to enable
    
    For more information about this error, try `rustc --explain E0658`.
    */
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

impl<C: Config> HasPrediction for C {
    type P = C::Prediction;
}

impl<C: Config> HasReward for C {
    type R = C::Reward;
}

impl<C: Config> HasExpectedFutureGain for C {
    type G = C::ExpectedFutureGain;
}

#[macro_export]
macro_rules! VisibleRewardTree {
    ($config:ty) => {
        $crate::visible_tree::VRewardTree<
            <$config as $crate::visible_tree::config::Config>::State,
            <$config as $crate::visible_tree::config::Config>::Path,
            <$config as $crate::visible_tree::config::Config>::RootData,
            <$config as $crate::visible_tree::config::Config>::StateData,
        >
    };
}

// I don't like the ergonomics -- to get the type, you need to qualify with <_ as _> stuff
// pub trait ToVisibleRewardTree {
//     type VRewardTree;
// }

// impl<C: Config> ToVisibleRewardTree for C {
//     type VRewardTree = super::VRewardTree<
//         C::State,
//         C::Path,
//         C::RootData,
//         C::StateData,
//     >;
// }
