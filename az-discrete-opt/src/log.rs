use crate::{iq_min_tree::{ActionMultiset, Transitions}, state::cost::Cost};

pub struct NextEpochRoot<S, C = f32> {
    next_root: S,
    cost: C,
    kind: CandidateKind,
    episodes: usize,
}

pub enum CandidateKind {
    NotReplacedDuringEpoch { minor_modifications: usize, epochs: usize },
    FoundDuringCurrentEpoch,
}

impl<S, C> NextEpochRoot<S, C> {
    pub fn new(next_root: S, cost: C) -> Self {
        Self {
            next_root,
            cost,
            kind: CandidateKind::NotReplacedDuringEpoch { minor_modifications: 0, epochs: 0 },
            episodes: 0,
        }
    }

    pub fn current_candidate(&self) -> RootCandidateData<C>
    where
        S: ShortForm,
        C: Cost + Clone,
    {
        let Self { next_root, cost, kind: _, episodes } = self;
        RootCandidateData {
            short_form: next_root.short_form(),
            cost: cost.clone(),
            episodes: *episodes,
        }
    }

    pub fn post_episode_update(&mut self, s_t: &S, c_t: &C) -> Option<RootCandidateData<C>>
    where
        S: ShortForm + Clone,
        C: Cost + Clone,
    {
        self.episodes += 1;
        match self.cost.cost().partial_cmp(&c_t.cost()).expect("costs must be comparable floats") {
            std::cmp::Ordering::Less | std::cmp::Ordering::Equal => None,
            std::cmp::Ordering::Greater => {
                let previous_candidate = self.current_candidate();
                self.next_root = s_t.clone();
                self.cost = c_t.clone();
                self.kind = CandidateKind::FoundDuringCurrentEpoch;
                self.episodes = 0;
                Some(previous_candidate)
            }
        }
    }
}

pub trait ShortForm {
    fn short_form(&self) -> String;
}

pub struct RootCandidateData<C = f32> {
    short_form: String,
    cost: C,
    episodes: usize,
}

impl<C> RootCandidateData<C> {
    pub fn short_form(&self) -> &str {
        &self.short_form
    }
    
    pub fn cost(&self) -> &C {
        &self.cost
    }

    pub fn duration(&self) -> usize {
        self.episodes
    }
}

impl<C: core::fmt::Debug> core::fmt::Debug for RootCandidateData<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            short_form,
            cost,
            episodes,
        } = self;
        f.debug_map()
            .entry(&"short_form", short_form)
            .entry(&"cost", cost)
            .entry(&"duration", episodes)
            .finish()
    }
}

#[derive(Default)]
pub struct BasicLog {
    path: ActionMultiset,
    gain: f32,
}

impl BasicLog {
    pub fn new() -> Self {
        Self {
            path: ActionMultiset::empty(),
            gain: 0.0,
        }
    }

    pub fn path(&self) -> &ActionMultiset {
        &self.path
    }

    // pub fn par_new_logs<const BATCH: usize>() -> [Self; BATCH] {
    //     let mut logs: [MaybeUninit<Self>; BATCH] = MaybeUninit::uninit_array();
    //     logs.par_iter_mut().for_each(|l| {
    //         l.write(Self::new());
    //     });
    //     unsafe { MaybeUninit::array_assume_init(logs) }
    // }

    pub fn update(&mut self, transitions: &Transitions) {
        let Self {
            path: logged_path,
            gain: logged_gain,
        } = self;
        let Transitions {
            first_action: _,
            first_reward,
            transitions,
            end,
        } = transitions;
        let mut gain = *first_reward;
        use core::cmp::Ordering;
        transitions.iter().for_each(|(path, _, reward)| {
            let gain_cmp = gain.total_cmp(logged_gain);
            let length_cmp = path.len().cmp(&logged_path.len());
            // prioritize gain, then length
            match (gain_cmp, length_cmp) {
                (Ordering::Greater, _) => {
                    *logged_gain = gain;
                    logged_path.clone_from(path);
                }
                (Ordering::Equal, Ordering::Greater) => {
                    *logged_gain = gain;
                    logged_path.clone_from(path);
                }
                _ => {}
            }
            gain += reward;
        });
        let end_path = end.path();
        let end_gain = end.gain();
        let gain_cmp = end_gain.total_cmp(logged_gain);
        let length_cmp = end_path.len().cmp(&logged_path.len());
        // prioritize gain, then length
        match (gain_cmp, length_cmp) {
            (Ordering::Greater, _) => {
                *logged_gain = end_gain;
                logged_path.clone_from(end_path);
            }
            (Ordering::Equal, Ordering::Greater) => {
                *logged_gain = end_gain;
                logged_path.clone_from(end_path);
            }
            _ => {}
        }
    }
}

// pub fn par_update_logs<const BATCH: usize>(
//     logs: &mut [BasicLog; BATCH],
//     transitions: &[Transitions; BATCH],
// ) {
//     let logs = logs.par_iter_mut();
//     let transitions = transitions.par_iter();
//     logs.zip_eq(transitions).for_each(|(l, t)| l.update(t));
// }
