use super::log::FinalStateData;

pub trait Config {
    type RootData;
    type StateData;
    type Prediction;
    type Path;
    type State;
    type Model;
    type Reward;
    type ExpectedFutureGain;
    type Log;
    // type Observation;
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

// pub trait HasObservation {
//     type O;
// }

// impl<C: Config> HasObservation for C {
//     type O = C::Observation;
// }

pub trait HasPrediction {
    type P;
}

pub trait HasLog {
    type L;
}

pub trait HasReward {
    type R;
}

pub trait HasModel {
    type M;
}

pub trait HasExpectedFutureGain {
    type G;
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

impl<C: Config> HasModel for C {
    type M = C::Model;
}

impl<C: Config> HasLog for C {
    type L = C::Log;
}

pub trait HasEndNode {
    type E;
}

impl<C: Config> HasEndNode for C {
    type E = FinalStateData<C::ExpectedFutureGain>;
}

#[macro_export]
macro_rules! VRewardTree {
    ($config:ty) => {
        $crate::visible_reward::VRewardTree<
            <$config as $crate::visible_reward::config::Config>::State,
            <$config as $crate::visible_reward::config::Config>::Path,
            <$config as $crate::visible_reward::config::Config>::RootData,
            <$config as $crate::visible_reward::config::Config>::StateData,
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
