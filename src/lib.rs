use std::collections::BTreeMap;

pub trait Path: Default + Ord {
    fn add_action(&mut self, action: usize);
}

pub struct VisibleRewardTree<S, P: Path, R: Reward, F: FutureGain<R>, const A_TOTAL: usize> {
    root: S,
    stats: BTreeMap<P, VisibleRewardStateStats<R, F>>
}

struct VisibleRewardStateStats<R: Reward, F: FutureGain<R>> {
    actions_taken: usize,
    action_stats: Vec<VisibleRewardActionStats<R, F>>
}

impl<R: Reward, F: FutureGain<R>> VisibleRewardStateStats<R, F> {
    fn best_action(&self) -> &VisibleRewardActionStats<R, F> {
        self.action_stats.first().unwrap()
    }

    fn update(&mut self, action: usize, future_reward: R, evaluation: f32) {
        let Self { actions_taken, action_stats } = self;
        action_stats[action].update(future_reward, evaluation);
        *actions_taken += 1;
        action_stats.sort_unstable_by(|a, b| {
            let u_first = a.upper_estimate(*actions_taken);
            let u_second = b.upper_estimate(*actions_taken);
            u_second.total_cmp(&u_first)
        });
    }
}

struct VisibleRewardActionStats<R: Reward, F: FutureGain<R>> {
    action: usize,
    frequency: usize,
    reward: R,
    probability: f32,
    total_future_gains: F
}

impl<R: Reward, F: FutureGain<R>> VisibleRewardActionStats<R, F> {
    fn update(&mut self, future_reward: R, evaluation: f32) {
        let Self { action: _, frequency, reward: _, probability: _, total_future_gains } = self;
        *frequency += 1;
        total_future_gains.add_reward(future_reward);
        total_future_gains.add_evaluation(evaluation);
    }

    fn upper_estimate(&self, actions: usize) -> f32 {
        let Self { action: _, frequency, reward, probability, total_future_gains } = self;
        let reward = reward.to_f32();
        let probability = *probability;
        let total_future_gains = total_future_gains.to_f32();
        reward + total_future_gains / *frequency as f32
        + C_PUCT * probability * (actions as f32).sqrt() / (1 + frequency) as f32
    }
}


const C_PUCT: f32 = 1.0;



impl<R: Reward, F: FutureGain<R>> VisibleRewardStateStats<R, F> {
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

pub trait Reward: core::ops::Add<Output = Self> + Sized {
    fn zero() -> Self;
    fn to_f32(&self) -> f32;
}

impl Reward for i32 {
    fn zero() -> Self {
        0
    }

    fn to_f32(&self) -> f32 {
        *self as f32
    }
}

pub trait FutureGain<R: Reward> {
    fn add_reward(&mut self, reward: R);
    fn add_evaluation(&mut self, evaluation: f32);
    fn zero() -> Self;
    fn to_f32(&self) -> f32;
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

    fn to_f32(&self) -> f32 {
        *self
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
        for i in 0..sims {
           let (rewards, results) = self.simulate();
            match results {
                SimulationResults::Leaf(paths) => {
                    dbg!(i, "leaf");
                    self.update_with_rewards(rewards, paths);
                },
                SimulationResults::New(paths, path, state) => {
                    dbg!(i, "new");
                    let (probabilities, evaluation) = evaluate.evaluate(&state);
                    let stats = VisibleRewardStateStats::new::<_, A_TOTAL>(&state, probabilities);
                    self.stats.insert(path, stats);
                    self.update_with_rewards_and_evaluation(paths, rewards, evaluation);
                },
            }
        }
    }

    fn simulate(&self) -> (Vec<(usize, R)>, SimulationResults<S, P>) {
        let mut state = self.root.clone();
        let mut path = P::default();
        let mut paths = Vec::new();
        let mut rewards = Vec::new();
        while !state.is_terminal() {
            if let Some(stats) = self.stats.get(&path) {
                let VisibleRewardActionStats { action, reward, ..} = stats.best_action();
                rewards.push((*action, reward.clone()));
                state.act(*action);
                paths.push(path.clone());
                path.add_action(*action);
            } else {
                return (rewards, SimulationResults::New(paths, path, state));
            }
        }
        return (rewards, SimulationResults::Leaf(paths));
    }

    
    
    fn update_with_rewards(&mut self, rewards: Vec<(usize, R)>, paths: Vec<P>) {
        self.update_with_rewards_and_evaluation(paths, rewards, 0.0);
    }

    fn update_with_rewards_and_evaluation(&mut self, paths: Vec<P>, rewards: Vec<(usize, R)>, end_evaluation: f32) {
        dbg!(rewards.iter().map(|(_, reward)| reward.to_f32()).collect::<Vec<_>>());
        assert_eq!(paths.len(), rewards.len());
        // let mut future_rewards = vec![];
        // let mut future_reward = R::zero();
        let mut rewards_iter = rewards.into_iter();
        let (mut future_rewards, mut future_reward, first_action, first_reward) = if let Some((action, reward)) = rewards_iter.next() {
            (vec![], R::zero(), action, reward)
        } else {
            return;
        };
        for (action, reward) in rewards_iter.rev() {
            future_reward = future_reward + reward.clone();
            future_rewards.push((action, future_reward.clone()));
        }
        future_rewards.reverse();
        let total_reward = first_reward.clone() + future_reward.clone();
        if total_reward.to_f32() > 9.9 {
            println!("action, reward = {}, {}", first_action, first_reward.to_f32());
            for (action, reward) in future_rewards.iter() {
                println!("action, future_reward = {}, {}", action, reward.to_f32());
            }
            panic!();
        }
        dbg!(future_rewards.iter().map(|(_, reward)| reward.to_f32()).collect::<Vec<_>>());
        paths.into_iter().zip(future_rewards.into_iter()).enumerate().for_each(|(i, (path, (action, future_reward)))| {
            let stats = self.stats.get_mut(&path).unwrap();
            stats.update(action, future_reward, end_evaluation);
        });
    }

    pub fn best(&self) -> S {
        todo!()
    }
}

pub enum SimulationResults<S, P> {
    Leaf(Vec<P>),
    New(Vec<P>, P, S),
}