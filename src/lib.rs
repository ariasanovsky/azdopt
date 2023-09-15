use std::collections::BTreeMap;

pub trait SearchPath: Default + Ord {}

pub struct VisibleRewardTree<S, P: SearchPath, R, F, const A_TOTAL: usize> {
    root: S,
    stats: BTreeMap<P, VisibleRewardStateStats<R, F>>
}

struct VisibleRewardStateStats<R, F> {
    actions_taken: usize,
    evaluation: f32,
    action_stats: Vec<VisibleRewardActionStats<R, F>>
}

struct VisibleRewardActionStats<R, F> {
    frequency: usize,
    reward: R,
    probability: f32,
    total_future_rewards: F
}

impl<R: Reward, F> VisibleRewardStateStats<R, F> {
    fn new<S: State<R>, E, const A_TOTAL: usize>(state: &S, evaluate: &E) -> Self
    where
        E: Evaluate<S>,
        F: FutureReward<R>,
    {
        let (probabilities, evaluation) = evaluate.evaluate(state);
        let action_stats = (0..A_TOTAL).filter_map(|action| {
            state.reward(action).map(|reward| {
                VisibleRewardActionStats {
                    frequency: 0,
                    reward,
                    probability: probabilities[action],
                    total_future_rewards: F::zero()
                }
            })
        }).collect();
        Self {
            actions_taken: 0,
            evaluation,
            action_stats
        }
    }
}

pub trait Evaluate<S> {
    fn evaluate(&self, state: &S) -> (Vec::<f32>, f32);
}

pub trait State<R: Reward> {
    fn reward(&self, action: usize) -> Option<R>;
}

pub trait Reward: core::ops::Add + Sized {}

impl Reward for i32 {}

pub trait FutureReward<R: Reward> {
    fn add_reward(&mut self, reward: R) -> Self;
    fn add_evaluation(&mut self, evaluation: f32) -> Self;
    fn zero() -> Self;
}

impl FutureReward<i32> for i32 {
    fn add_reward(&mut self, reward: i32) -> Self {
        todo!()
    }

    fn add_evaluation(&mut self, evaluation: f32) -> Self {
        todo!()
    }

    fn zero() -> Self {
        todo!()
    }
}

impl<S, P: SearchPath, R: Reward, F, const A_TOTAL: usize>
VisibleRewardTree<S, P, R, F, A_TOTAL>
where
    S: State<R>,
    F: FutureReward<R>,
{
    pub fn new<E: Evaluate<S>>(root: S, evaluate: &E) -> Self {
        let path = P::default();
        let root_stats = VisibleRewardStateStats::new::<_, _, A_TOTAL>(&root, evaluate);
        let mut stats = BTreeMap::new();
        stats.insert(path, root_stats);
        Self { root, stats }
    }

    pub fn simulate<E>(&mut self, sims: usize, depth: usize, evaluate: &E) {
        todo!()
    }

    pub fn best(&self) -> S {
        todo!()
    }
}