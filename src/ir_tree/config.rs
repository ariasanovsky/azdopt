use super::log::FinalStateData;

pub trait Config {
    type RootData;
    type StateData;
    type Prediction;
    type Path;
    type State;
    type Model;
    // type Reward;
    // type ExpectedFutureGain;
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

pub trait HasModel {
    type M;
}

impl<C: Config> HasPrediction for C {
    type P = C::Prediction;
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
    type E = FinalStateData;
}

#[macro_export]
macro_rules! VRewardTree {
    ($config:ty) => {
        $crate::ir_tree::IRTree<
            <$config as $crate::ir_tree::config::Config>::State,
            <$config as $crate::ir_tree::config::Config>::Path,
            <$config as $crate::ir_tree::config::Config>::RootData,
            <$config as $crate::ir_tree::config::Config>::StateData,
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
