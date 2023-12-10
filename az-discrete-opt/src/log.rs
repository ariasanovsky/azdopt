use crate::iq_min_tree::{ActionMultiset, Transitions};

// pub struct NextEpochRoot<S, C> {
//     next_root: S,
//     cost: C,
//     kind: CandidateKind,
//     episodes: usize,
// }

// pub enum CandidateKind {
//     NotReplacedDuringEpoch(Stagnation),
//     FoundDuringCurrentEpoch,
// }

// #[derive(Clone, Debug)]
// pub struct Stagnation {
//     pub minor_modifications: usize,
//     pub epochs: usize,
// }

// impl Stagnation {
//     pub fn new() -> Self {
//         Self {
//             minor_modifications: 0,
//             epochs: 0,
//         }
//     }
// }

// impl<S, C> NextEpochRoot<S, C> {
//     pub fn new(next_root: S, cost: C) -> Self {
//         Self {
//             next_root,
//             cost,
//             kind: CandidateKind::NotReplacedDuringEpoch(Stagnation { minor_modifications: 0, epochs: 0 }),
//             episodes: 0,
//         }
//     }

//     pub fn current_candidate(&self) -> ArgminData<C>
//     where
//         S: ShortForm,
//         C: Cost + Clone,
//     {
//         let Self { next_root, cost, kind: _, episodes } = self;
//         ArgminData {
//             short_form: next_root.short_form(),
//             cost: cost.clone(),
//             episodes: *episodes,
//         }
//     }

//     pub fn current_root(&self) -> &S {
//         &self.next_root
//     }

//     pub fn post_episode_update(&mut self, s_t: &S, c_t: &C) -> Option<ArgminData<C>>
//     where
//         S: ShortForm + Clone,
//         C: Cost + Clone,
//     {
//         self.episodes += 1;
//         match self.cost.evaluate().partial_cmp(&c_t.evaluate()).expect("costs must be comparable floats") {
//             std::cmp::Ordering::Less | std::cmp::Ordering::Equal => None,
//             std::cmp::Ordering::Greater => {
//                 // dbg!();
//                 let previous_candidate = self.current_candidate();
//                 self.next_root.clone_from(s_t);
//                 self.cost.clone_from(c_t);
//                 self.kind = CandidateKind::FoundDuringCurrentEpoch;
//                 self.episodes = 0;
//                 Some(previous_candidate)
//             }
//         }
//     }

//     pub fn end_epoch(&mut self) -> (ArgminData<C>, Option<&Stagnation>)
//     where
//         S: ShortForm,
//         C: Cost + Clone,
//     {
//         let candidate = self.current_candidate();
//         let Self { next_root: _, cost: _, kind, episodes } = self;
//         match kind {
//             CandidateKind::NotReplacedDuringEpoch(stag) => {
//                 stag.epochs += 1;
//                 *episodes = 0;
//                 (candidate, Some(stag))
//             },
//             CandidateKind::FoundDuringCurrentEpoch => {
//                 *kind = CandidateKind::NotReplacedDuringEpoch(Stagnation::new());
//                 *episodes = 0;
//                 (candidate, None)
//             }
//         }
//     }

//     pub fn make_minor_modification(&mut self, f: impl Fn(&mut S)) -> bool {
//         match &mut self.kind {
//             CandidateKind::NotReplacedDuringEpoch(stag) => {
//                 f(&mut self.next_root);
//                 stag.minor_modifications += 1;
//                 true
//             },
//             CandidateKind::FoundDuringCurrentEpoch => false,
//         }
//     }
// }

pub trait ShortForm {
    fn short_form(&self) -> String;
}

pub struct ArgminData<C> {
    short_form: String,
    cost: C,
    episode: usize,
    epoch: usize,
}

impl<C> ArgminData<C> {
    pub fn new(s: &impl ShortForm, cost: C, episode: usize, epoch: usize) -> Self {
        Self {
            short_form: s.short_form(),
            cost,
            episode,
            epoch,
        }
    }

    pub fn short_form(&self) -> &str {
        &self.short_form
    }

    pub fn cost(&self) -> &C {
        &self.cost
    }
}

impl<C: core::fmt::Debug> core::fmt::Debug for ArgminData<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self {
            short_form,
            cost,
            episode,
            epoch,
        } = self;
        f.debug_map()
            .entry(&"short_form", short_form)
            .entry(&"cost", cost)
            .entry(&"episode", episode)
            .entry(&"epoch", epoch)
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
