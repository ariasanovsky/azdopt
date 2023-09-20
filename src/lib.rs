use std::collections::BTreeMap;

pub mod visible_reward;

pub trait Path: Default + Ord {
    fn add_action(&mut self, action: usize);
}

pub struct VisibleRewardTree<C: VisibleRewardSearchConfig, M: VisibleRewardModelConfig> {
    root: C::S,
    stats: BTreeMap<C::P, VisibleRewardStateStats<C::F>>,
    phantom: std::marker::PhantomData<M>
}

struct VisibleRewardStateStats<F: FutureGain> {
    actions_taken: usize,
    action_stats: Vec<VisibleRewardActionStats<F>>
}

impl<F: FutureGain> VisibleRewardStateStats<F> {
    fn best_action(&self) -> &VisibleRewardActionStats<F> {
        self.action_stats.first().unwrap()
    }

    fn update<M: VisibleRewardModelConfig>(&mut self, action: usize, future_reward: F::R, evaluation: f32) {
        let Self { actions_taken, action_stats } = self;
        action_stats[action].update(future_reward, evaluation);
        *actions_taken += 1;
        action_stats.sort_unstable_by(|a, b| {
            let u_first = a.upper_estimate::<M>(*actions_taken);
            let u_second = b.upper_estimate::<M>(*actions_taken);
            u_second.total_cmp(&u_first)
        });
    }
}

struct VisibleRewardActionStats<F: FutureGain> {
    action: usize,
    frequency: usize,
    reward: F::R,
    probability: f32,
    total_future_gains: F
}

impl<F: FutureGain> VisibleRewardActionStats<F> {
    fn update(&mut self, future_reward: F::R, evaluation: f32) {
        let Self { action: _, frequency, reward: _, probability: _, total_future_gains } = self;
        *frequency += 1;
        total_future_gains.add_reward(future_reward);
        total_future_gains.add_evaluation(evaluation);
    }

    fn upper_estimate<M: VisibleRewardModelConfig>(&self, actions: usize) -> f32 {
        let Self { action: _, frequency, reward, probability, total_future_gains } = self;
        let reward = reward.to_f32();
        let probability = *probability;
        let total_future_gains = total_future_gains.to_f32();
        M::action_estimate(reward, probability, actions, *frequency, total_future_gains)
        // reward + total_future_gains / *frequency as f32
        // + C_PUCT * probability * (actions as f32).sqrt() / (1 + frequency) as f32
    }
}


impl<F: FutureGain> VisibleRewardStateStats<F> {
    fn new<S: State, const A_TOTAL: usize, M: VisibleRewardModelConfig>(state: &S, probabilities: Vec<f32>) -> Self
    where
        F: FutureGain<R = S::R>,
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
            let u_first = M::new_state_action_estimate(a.reward.to_f32(), a.probability);
            let u_second = M::new_state_action_estimate(b.reward.to_f32(), b.probability);
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

pub trait State {
    type R: Reward;
    fn reward(&self, action: usize) -> Option<Self::R>;
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

pub trait FutureGain {
    type R: Reward;
    fn add_reward(&mut self, reward: Self::R);
    fn add_evaluation(&mut self, evaluation: f32);
    fn zero() -> Self;
    fn to_f32(&self) -> f32;
}

impl FutureGain for f32 {
    type R = i32;
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

pub trait VisibleRewardSearchConfig {
    type S: State + Clone;
    type P: Path + Clone;
    type F: FutureGain<R = <Self::S as State>::R>;
}

pub trait VisibleRewardModelConfig {
    fn new_state_action_estimate(reward: f32, probability: f32) -> f32 {
        const C_PUCT: f32 = 1.0;
        reward + C_PUCT * probability
    }
    
    fn action_estimate(reward: f32, probability: f32, actions: usize, frequency: usize, total_future_gains: f32) -> f32 {
        const C_PUCT: f32 = 1.0;
        reward + total_future_gains / frequency as f32
        + C_PUCT * probability * (actions as f32).sqrt() / (1 + frequency) as f32
    }
}

impl<C: VisibleRewardSearchConfig, M: VisibleRewardModelConfig> VisibleRewardTree<C, M> {
    pub fn new(root: C::S) -> Self {
        Self { root, stats: Default::default(), phantom: Default::default() }
    }

    pub fn simulate_and_update<E: Evaluate<C::S>, const A_TOTAL: usize>(&mut self, sims: usize, evaluate: &E)
    where
        C::P: Clone,
        <C::S as State>::R: Clone,
    {
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
                    let stats = VisibleRewardStateStats::new::<_, A_TOTAL, M>(&state, probabilities);
                    self.stats.insert(path, stats);
                    self.update_with_rewards_and_evaluation(paths, rewards, evaluation);
                },
            }
        }
    }

    fn simulate(&self) -> (Vec<(usize, <C::S as State>::R)>, SimulationResults<C::S, C::P>)
    where
        <C::S as State>::R: Clone,
    {
        let mut state = self.root.clone();
        let mut path = C::P::default();
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

    
    
    fn update_with_rewards(&mut self, rewards: Vec<(usize, <C::S as State>::R)>, paths: Vec<C::P>)
    where
        <C::S as State>::R: Clone,
    {
        self.update_with_rewards_and_evaluation(paths, rewards, 0.0);
    }

    fn update_with_rewards_and_evaluation(&mut self, paths: Vec<C::P>, rewards: Vec<(usize, <C::S as State>::R)>, end_evaluation: f32)
    where
        <C::S as State>::R: Clone,
        M: VisibleRewardModelConfig,
    {
        dbg!(rewards.iter().map(|(_, reward)| reward.to_f32()).collect::<Vec<_>>());
        assert_eq!(paths.len(), rewards.len());
        // let mut future_rewards = vec![];
        // let mut future_reward = R::zero();
        let mut rewards_iter = rewards.into_iter();
        let (mut future_rewards, mut future_reward, first_action, first_reward) = if let Some((action, reward)) = rewards_iter.next() {
            (vec![], <C::S as State>::R::zero(), action, reward)
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
            stats.update::<M>(action, future_reward, end_evaluation);
        });
    }

    pub fn best(&self) -> C::S {
        todo!()
    }
}

pub enum SimulationResults<S, P> {
    Leaf(Vec<P>),
    New(Vec<P>, P, S),
}