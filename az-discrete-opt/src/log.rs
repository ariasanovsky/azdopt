use core::mem::MaybeUninit;

use rayon::prelude::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
    IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{
    iq_min_tree::{ActionMultiset, Transitions},
    state::StateNode,
};

pub struct SimpleRootLog<S, C = f32> {
    next_root: S,
    root_cost: C,
    duration: usize,
    stagnation: Option<usize>,
    short_data: Vec<ShortRootData<C>>,
}

impl<S, C> SimpleRootLog<S, C> {
    pub fn new(cost: &C, s: &S) -> Self
    where
        S: Clone,
        C: Clone,
    {
        Self {
            next_root: s.clone(),
            root_cost: cost.clone(),
            duration: 0,
            stagnation: Some(0),
            short_data: vec![],
        }
    }

    pub fn stagnation(&self) -> Option<usize> {
        self.stagnation
    }

    pub fn increment_stagnation(&mut self) {
        if let Some(s) = &mut self.stagnation {
            *s += 1;
        }
        // self.stagnation.as_mut().map(|s| *s += 1);
    }

    // todo! make private
    pub fn zero_stagnation(&mut self) {
        self.stagnation = Some(0);
    }

    pub fn empty_stagnation(&mut self) {
        self.stagnation = None;
    }

    pub fn par_new_logs<const BATCH: usize>(s_t: &[S; BATCH], costs: &[C; BATCH]) -> [Self; BATCH]
    where
        S: Sync + Clone,
        C: Sync + Clone,
        Self: Send,
    {
        let mut logs: [MaybeUninit<Self>; BATCH] = MaybeUninit::uninit_array();
        (&mut logs, s_t, costs)
            .into_par_iter()
            .for_each(|(l, s, cost)| {
                l.write(Self::new(cost, s));
            });
        unsafe { MaybeUninit::array_assume_init(logs) }
    }

    pub fn update(&mut self, s_t: &S, c_t: &C)
    where
        S: Clone + ShortForm,
        C: Clone + crate::state::cost::Cost<f32>,
    {
        let Self {
            next_root,
            root_cost,
            duration,
            stagnation,
            short_data,
        } = self;
        // todo!("what to do with stagnation?");
        // todo! tiebreakers
        // dbg!(c_t);
        if c_t.cost().lt(&root_cost.cost()) {
            let old_short_data = ShortRootData {
                short_form: next_root.short_form(),
                cost: root_cost.clone(),
                duration: *duration,
            };
            short_data.push(old_short_data);
            root_cost.clone_from(c_t);
            next_root.clone_from(s_t);
            *duration = 1;
            *stagnation = None;
            // println!("new best state: {c_star}");
        } else {
            *duration += 1;
        }
    }

    pub fn reset_root(&mut self, s: &S, cost: &C)
    where
        S: Clone + ShortForm,
        C: Clone,
    {
        let short_data = ShortRootData {
            short_form: self.next_root.short_form(),
            cost: self.root_cost.clone(),
            duration: self.duration,
        };
        self.short_data.push(short_data);
        self.next_root.clone_from(s);
        self.root_cost.clone_from(cost);
        self.zero_stagnation();
        self.zero_duration();
        // todo!("what to do with the logged data?")
    }

    pub fn next_root(&self) -> &S {
        &self.next_root
    }

    pub fn next_root_mut(&mut self) -> &mut S {
        &mut self.next_root
    }

    pub fn root_cost(&self) -> &C {
        &self.root_cost
    }

    fn zero_duration(&mut self) {
        self.duration = 0;
    }

    pub fn empty_root_data(&mut self, other: &mut Vec<ShortRootData<C>>)
    where
        S: ShortForm,
        C: Clone,
    {
        let short_data = ShortRootData {
            short_form: self.next_root.short_form(),
            cost: self.root_cost.clone(),
            duration: self.duration,
        };

        self.zero_duration();

        other.extend(
            self.short_data
                .drain(..)
                .chain(core::iter::once(short_data)),
        );
    }
}

pub trait ShortForm {
    fn short_form(&self) -> String;
}

impl<T: ShortForm> ShortForm for StateNode<T> {
    fn short_form(&self) -> String {
        let Self {
            state,
            time: _,
            prohibited_actions: _,
        } = &self;
        state.short_form()
    }
}

// #[derive(Debug)]
pub struct ShortRootData<C = f32> {
    short_form: String,
    cost: C,
    duration: usize,
}

impl<C: core::fmt::Debug> core::fmt::Debug for ShortRootData<C> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let Self { short_form, cost, duration} = self;
        f.debug_map()
            .entry(&"short_form", short_form)
            .entry(&"cost", cost)
            .entry(&"duration", duration)
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

    pub fn par_new_logs<const BATCH: usize>() -> [Self; BATCH] {
        let mut logs: [MaybeUninit<Self>; BATCH] = MaybeUninit::uninit_array();
        logs.par_iter_mut().for_each(|l| {
            l.write(Self::new());
        });
        unsafe { MaybeUninit::array_assume_init(logs) }
    }

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

pub fn par_update_logs<const BATCH: usize>(
    logs: &mut [BasicLog; BATCH],
    transitions: &[Transitions; BATCH],
) {
    let logs = logs.par_iter_mut();
    let transitions = transitions.par_iter();
    logs.zip_eq(transitions).for_each(|(l, t)| l.update(t));
}
