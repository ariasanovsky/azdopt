use std::collections::BTreeMap;

pub trait Path: Default + Ord {
    fn add_action(&mut self, action: usize);
}

pub struct VisibleRewardTree<S, P: Path, R, F, const A_TOTAL: usize> {
    root: S,
    stats: BTreeMap<P, VisibleRewardStateStats<R, F>>
}

struct VisibleRewardStateStats<R, F> {
    actions_taken: usize,
    action_stats: Vec<VisibleRewardActionStats<R, F>>
}

impl<R, F> VisibleRewardStateStats<R, F> {
    fn best_action(&self) -> &VisibleRewardActionStats<R, F> {
        todo!()
    }
}

struct VisibleRewardActionStats<R, F> {
    action: usize,
    frequency: usize,
    reward: R,
    probability: f32,
    total_future_gains: F
}


const C_PUCT: f32 = 1.0;



impl<R: Reward, F> VisibleRewardStateStats<R, F> {
    fn new<S: State<R>, const A_TOTAL: usize>(state: &S, probabilities: Vec<f32>) -> Self
    where
        F: FutureGain<R>,
    {
        let mut action_stats: Vec<_> = (0..A_TOTAL).filter_map(|action| {
            state.reward(action).map(|reward| {
                VisibleRewardActionStats {
                    action,
                    frequency: 0,
                    reward,
                    probability: probabilities[action],
                    total_future_gains: F::zero()
                }
            })
        }).collect();
        let probability_sum: f32 = action_stats.iter().map(|action_stats| action_stats.probability).sum();
        action_stats.iter_mut().for_each(|action_stats| {
            action_stats.probability /= probability_sum;
        });
        action_stats.sort_unstable_by(|a, b| {
            let u_first = a.reward.to_f32() + C_PUCT * a.probability;
            let u_second = b.reward.to_f32() + C_PUCT * b.probability;
            u_second.total_cmp(&u_first)
        });
        Self {
            actions_taken: 0,
            action_stats
        }
    }
}

pub trait Evaluate<S> {
    fn evaluate(&self, state: &S) -> (Vec::<f32>, f32);
}

pub trait State<R: Reward> {
    fn reward(&self, action: usize) -> Option<R>;
    fn is_terminal(&self) -> bool;
    fn act(&mut self, action: usize);
}

pub trait Reward: core::ops::Add + Sized {
    fn to_f32(&self) -> f32;
}

impl Reward for i32 {
    fn to_f32(&self) -> f32 {
        *self as f32
    }
}

pub trait FutureGain<R: Reward> {
    fn add_reward(&mut self, reward: R);
    fn add_evaluation(&mut self, evaluation: f32);
    fn zero() -> Self;
}

impl FutureGain<i32> for f32 {
    fn add_reward(&mut self, reward: i32) {
        *self += reward as f32;
    }

    fn add_evaluation(&mut self, evaluation: f32) {
        *self += evaluation;
    }

    fn zero() -> Self {
        0.0
    }
}

impl<S: Clone, P: Path + Clone, R: Reward + Clone, F, const A_TOTAL: usize>
VisibleRewardTree<S, P, R, F, A_TOTAL>
where
    S: State<R>,
    F: FutureGain<R>,
{
    pub fn new(root: S) -> Self {
        Self { root, stats: Default::default() }
    }

    pub fn simulate_and_update<E: Evaluate<S>>(&mut self, sims: usize, evaluate: &E) {
        for _ in 0..sims {
           let (rewards, results) = self.simulate();
            match results {
                SimulationResults::Leaf(paths) => {
                    self.update_with_rewards(rewards, paths);
                },
                SimulationResults::New(paths, path, state) => {
                    let (probabilities, evaluation) = evaluate.evaluate(&state);
                    let stats = VisibleRewardStateStats::new::<_, A_TOTAL>(&state, probabilities);
                    self.stats.insert(path, stats);
                    self.update_with_rewards_and_evaluation(paths, rewards, evaluation);
                },
            }
        }
    }

    fn simulate(&self) -> (Vec<R>, SimulationResults<S, P>) {
        let mut state = self.root.clone();
        let mut path = P::default();
        let mut paths = Vec::new();
        let mut rewards = Vec::new();
        while !state.is_terminal() {
            if let Some(stats) = self.stats.get(&path) {
                let VisibleRewardActionStats { action, reward, ..} = stats.best_action();
                rewards.push(reward.clone());
                state.act(*action);
                paths.push(path.clone());
                path.add_action(*action);
            } else {
                return (rewards, SimulationResults::New(paths, path, state));
            }
        }
        return (rewards, SimulationResults::Leaf(paths));
    }

    
    
    fn update_with_rewards(&mut self, rewards: Vec<R>, paths: Vec<P>) {
        let mut state = self.root.clone();
        todo!()
    }

    fn update_with_rewards_and_evaluation(&mut self, paths: Vec<P>, rewards: Vec<R>, evaluation: f32) {
        todo!()
    }

    pub fn best(&self) -> S {
        todo!()
    }
}

pub enum SimulationResults<S, P> {
    Leaf(Vec<P>),
    New(Vec<P>, P, S),
}