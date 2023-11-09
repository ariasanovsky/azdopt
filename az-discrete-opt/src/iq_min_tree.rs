use std::collections::BTreeMap;

#[derive(Clone, PartialEq, PartialOrd, Eq, Ord)]
pub struct ActionsTaken {
    actions_taken: Vec<usize>,
}

impl ActionsTaken {
    pub fn empty() -> Self {
        Self {
            actions_taken: vec![],
        }
    }

    pub(crate) fn clear(&mut self) {
        self.actions_taken.clear();
    }

    pub fn new(a_1: usize) -> Self {
        Self {
            actions_taken: vec![a_1],
        }
    }

    pub fn push(&mut self, action: usize) {
        self.actions_taken.push(action);
        self.actions_taken.sort_unstable();
    }

    pub fn len(&self) -> usize {
        self.actions_taken.len()
    }

    pub fn is_empty(&self) -> bool {
        self.actions_taken.is_empty()
    }

    pub fn actions(&self) -> &[usize] {
        &self.actions_taken
    }
}

// todo! move the root & root_cost outside the tree to batch-sized arrays
pub struct IQMinTree {
    root_data: IQStateData,
    data: BTreeMap<ActionsTaken, IQStateData>,
}

// todo! refactor with
//pub enum IRStateData {
//     Active { frequency: usize, Vec<IRActionData> },
//     FullyExplored { best_final_value: f32 }
// }
pub struct IQStateData {
    frequency: usize,
    actions: Vec<IQActionData>,
}

/* todo! make family of trees:
    * IQMinTree
      * (I)nitial cost is calculated when rooting a tree
      * (Q)quickly calculate rewards when new states are encountered
      * thus, we use q(s, a) = r(s, a) + q'(s, a)
    * IAMinTree
      * (I)mmediately cost calculated when rooting a tree
      * (A)ction rewards are calculated during the transition between states only
      * we still use q(s, a) = r(s, a) + q'(s, a)
        * but we initialize r(s, a) = 0 when s is added
        * and we update r(s, a) when visiting (s, a) for the first time
        * NOTE: this motivates replacing frequency with Option<NonzeroUsize>
    * INTMinTree (similar to IAMinTree)
      * (I)nitial cost is calculated when rooting a tree
      * (N)ew states added also get their costs calculated and stored
      * (T)erminal states also get their costs calculated and stored
        * we still use q(s, a) = r(s, a) + q'(s, a)
          * but we initialize r(s, a) = 0 when s is added
          * and we update r(s, a) when visiting (s, a) for the first time
          * NOTE: this motivates replacing frequency with Option<NonzeroUsize>
    * TMinTree
      * (T)erminal states get their costs calculated and stored
        * only terminal states 
*/

/* todo! what to do when (s, a) transitions to a terminal state?
  * in future visits to s, we need not consider a
  * thus we can remove a from the action space
  * if s has no actions, it is `fully explored`
  * some refactors to consider:
    * best_action returns an enum:
      * action (and reward),
      * fully_explored (with final gain)
      * NOTE: enum with variants Vec<T> (T not ZST?) & f32 has size 3 * 8
*/

// todo! refactor into a pair of values
pub struct IQActionData {
    // not mut
    action: usize,
    probability: f32,
    reward: f32,
    // mut
    frequency: usize,
    q_minus_r_sum: f32,
    upper_estimate: f32,
}

// todo! refactor without the globals, haven't thought too hard about it, low priority
const C_PUCT: f32 = 5.0;
const C_PUCT_0: f32 = 5.0;

impl IQActionData {
    fn new(action: usize, reward: f32, probability: f32) -> Self {
        let q = reward;
        // const C_PUCT_0: f32 = 30.0;
        let noise = 1.0;
        let u = q + C_PUCT_0 * probability * noise;
        Self {
            action,
            probability,
            reward,
            frequency: 0,
            q_minus_r_sum: 0.0,
            upper_estimate: u,
        }
    }

    fn update_future_reward(&mut self, approximate_gain_to_terminal: f32) {
        let Self {
            action: _,
            probability: _,
            reward: _,
            frequency,
            q_minus_r_sum: future_reward_sum,
            upper_estimate: _,
        } = self;
        *frequency += 1;
        *future_reward_sum += approximate_gain_to_terminal;
    }

    fn update_upper_estimate(&mut self, frequency: usize) {
        let Self {
            action: _,
            probability,
            reward,
            frequency: action_frequency,
            q_minus_r_sum: future_reward_sum,
            upper_estimate,
        } = self;
        let q = *reward + *future_reward_sum / frequency as f32;
        // const C_PUCT: f32 = 30.0;
        let noise = (frequency as f32).sqrt() / (1.0 + *action_frequency as f32);
        let u = q + C_PUCT * *probability * noise;
        *upper_estimate = u;
    }
}

impl IQStateData {
    fn new(predicted_probs: &[f32], action_rewards: &[(usize, f32)]) -> Self {
        let actions = action_rewards
            .iter()
            .map(|(a, r)| {
                let p = predicted_probs.get(*a).unwrap();
                IQActionData::new(*a, *r, *p)
            })
            .collect();
        Self {
            frequency: 0,
            actions,
        }
    }

    fn best_action(&self) -> (usize, f32) {
        self.actions.first().map(|a| (a.action, a.reward)).unwrap()
    }

    fn update_future_reward(&mut self, action: usize, approximate_gain_to_terminal: f32) {
        let Self { frequency, actions } = self;
        *frequency += 1;
        let action_data = actions.iter_mut().find(|a| a.action == action).unwrap();
        action_data.update_future_reward(approximate_gain_to_terminal);
        actions
            .iter_mut()
            .for_each(|a| a.update_upper_estimate(*frequency));
        actions.sort_unstable_by(|a, b| b.upper_estimate.total_cmp(&a.upper_estimate));
    }
}

impl IQMinTree {
    pub fn new(rewards: &[(usize, f32)], probability_predictions: &[f32]) -> Self {
        let mut actions: Vec<_> = rewards
            .iter()
            .map(|(i, r)| {
                let p = *probability_predictions.get(*i).unwrap();
                // todo? use a method here instead
                // const C_PUCT: f32 = 30.0;
                let u = r + C_PUCT * p;
                IQActionData {
                    action: *i,
                    probability: p,
                    reward: *r,
                    frequency: 0,
                    q_minus_r_sum: 0.0,
                    upper_estimate: u,
                }
            })
            .collect();
        actions.sort_unstable_by(|a, b| b.upper_estimate.total_cmp(&a.upper_estimate));
        let root_data = IQStateData {
            frequency: 0,
            actions,
        };
        Self {
            root_data,
            data: Default::default(),
        }
    }

    // todo! refactor so that transitions instead hold &mut's to the values
    pub fn simulate_once<const STATE: usize, S>(&self, root: &mut S) -> Transitions
    where
        S: IQState<STATE>,
    {
        let Self { root_data, data } = self;
        let state = root;
        let (first_action, first_reward) = root_data.best_action();
        state.act(first_action);
        let mut state_path = ActionsTaken::new(first_action);
        let mut transitions: Vec<(ActionsTaken, usize, f32)> = vec![];
        let mut gain = first_reward;
        while !state.is_terminal() {
            if let Some(data) = data.get(&state_path) {
                let (action, reward) = data.best_action();
                gain += reward;
                state.act(action);
                transitions.push((state_path.clone(), action, reward));
                state_path.push(action);
            } else {
                // new state
                return Transitions {
                    first_action,
                    first_reward,
                    transitions,
                    end: SearchEnd::New {
                        end: state_path,
                        gain,
                    },
                };
            }
        }
        Transitions {
            first_action,
            first_reward,
            transitions,
            end: SearchEnd::Terminal {
                end: state_path,
                gain,
            },
        }
    }

    // todo! refactor without the `&mut self` to update the &mut values in-place
    pub fn update(&mut self, transitions: &Transitions, gains: &[f32]) {
        let Self { root_data, data } = self;
        let Transitions {
            first_action,
            first_reward,
            transitions,
            end: new_path,
        } = transitions;
        debug_assert_eq!(gains.len(), 1);
        let mut approximate_gain_to_terminal = match new_path {
            SearchEnd::Terminal { .. } => 0.0f32,
            SearchEnd::New { .. } => gains.first().unwrap().max(0.0),
        };
        // values compliant with https://github.com/ariasanovsky/azdopt/issues/11
        transitions.iter().rev().for_each(|(path, action, reward)| {
            let state_data = data.get_mut(path).unwrap();
            approximate_gain_to_terminal = approximate_gain_to_terminal.max(0.0);
            state_data.update_future_reward(*action, approximate_gain_to_terminal);
            approximate_gain_to_terminal += reward;
        });
        approximate_gain_to_terminal = approximate_gain_to_terminal.max(0.0);
        root_data.update_future_reward(*first_action, approximate_gain_to_terminal);
        approximate_gain_to_terminal += first_reward;
    }

    // todo! this only uses the end path
    pub fn insert(
        &mut self,
        transitions: &Transitions,
        end_state_rewards: &[(usize, f32)],
        probs: &[f32],
    ) {
        let Self { root_data: _, data } = self;
        let Transitions {
            first_action: _,
            first_reward: _,
            transitions: _,
            end: new_path,
        } = transitions;
        if let SearchEnd::New { end: new_path, .. } = new_path {
            let state_data = IQStateData::new(probs, end_state_rewards);
            data.insert(new_path.clone(), state_data);
        }
    }

    // todo! this is only a method on `root_data`
    pub fn observations<const ACTION: usize>(&self) -> Vec<f32> {
        let Self { root_data, data: _ } = self;
        let IQStateData { frequency, actions } = root_data;
        let mut gain_sum = 0.0;
        let mut observations = vec![0.0; ACTION + 1];
        actions.iter().for_each(|a| {
            let IQActionData {
                action,
                probability: _,
                reward,
                frequency: action_frequency,
                q_minus_r_sum: future_reward_sum,
                upper_estimate: _,
            } = a;
            // todo! should there be a max(0.0) here?
            gain_sum += *reward;
            gain_sum += *future_reward_sum;
            *observations.get_mut(*action).unwrap() = *action_frequency as f32 / *frequency as f32;
        });
        *observations.last_mut().unwrap() = gain_sum / *frequency as f32;
        observations
    }
}

pub trait IQState<const STATE: usize> {
    const ACTION: usize;
    fn action_rewards(&self) -> Vec<(usize, f32)>;
    fn act(&mut self, action: usize);
    fn is_terminal(&self) -> bool;
    fn apply(&mut self, actions: &ActionsTaken) {
        actions.actions_taken.iter().for_each(|a| self.act(*a));
    }
    fn reset(&mut self, time: usize);
    fn to_vec(&self) -> [f32; STATE];
}

pub enum SearchEnd {
    Terminal { end: ActionsTaken, gain: f32 },
    New { end: ActionsTaken, gain: f32 },
}

impl SearchEnd {
    pub fn gain(&self) -> f32 {
        match self {
            Self::Terminal { gain, .. } => *gain,
            Self::New { gain, .. } => *gain,
        }
    }

    pub fn path(&self) -> &ActionsTaken {
        match self {
            Self::Terminal { end, .. } => end,
            Self::New { end, .. } => end,
        }
    }
}

// todo! refactor with &mut's to the tree's values instead of vecs of cloned vecs
pub struct Transitions {
    pub(crate) first_action: usize,
    pub(crate) first_reward: f32,
    pub(crate) transitions: Vec<(ActionsTaken, usize, f32)>,
    pub(crate) end: SearchEnd,
}

impl Transitions {
    pub fn costs(&self, root_cost: f32) -> Vec<f32> {
        let mut costs = vec![root_cost];
        let mut cost = root_cost - self.first_reward;
        costs.push(cost);
        let rewards = self.transitions.iter().map(|(_, _, r)| {
            cost -= r;
            cost
        });
        costs.extend(rewards);
        costs
    }

    pub fn end(&self) -> &SearchEnd {
        &self.end
    }
}
