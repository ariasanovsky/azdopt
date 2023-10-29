use core::mem::MaybeUninit;

use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::{iq_min_tree::{ActionsTaken, Transitions}, int_min_tree::INTTransitions, state::Cost};

pub struct CostLog {
    p_star: ActionsTaken,
    c_star: f32,
}

impl CostLog {
    pub fn new(cost: f32) -> Self {
        Self {
            p_star: ActionsTaken::empty(),
            c_star: cost,
        }
    }

    pub fn path(&self) -> &ActionsTaken {
        &self.p_star
    }

    pub fn par_new_logs<const BATCH: usize, S: Cost + Sync>(s_t: &[S; BATCH]) -> [Self; BATCH] {
        let states = s_t.par_iter();
        let mut logs: [MaybeUninit<Self>; BATCH] = MaybeUninit::uninit_array();
        logs.par_iter_mut().zip_eq(states).for_each(|(l, s)| {
            l.write(Self::new(s.cost()));
        });
        unsafe { MaybeUninit::array_assume_init(logs) }
    }

    pub fn update(&mut self, transitions: &INTTransitions, c_t: f32) {
        let Self { 
            p_star,
            c_star,
        } = self;
        let p_t = transitions.last_path();
        let cost_cmp = c_t.total_cmp(c_star);
        let length_cmp = p_t.len().cmp(&p_star.len());
        // prioritize cost, then length
        match (cost_cmp, length_cmp) {
            (core::cmp::Ordering::Less, _) => {
                *c_star = c_t;
                p_star.clone_from(p_t);
            }
            (core::cmp::Ordering::Equal, core::cmp::Ordering::Less) => {
                *c_star = c_t;
                p_star.clone_from(p_t);
            }
            _ => {}
        }
    }
}

pub struct BasicLog {
    path: ActionsTaken,
    gain: f32,
}

impl BasicLog {
    pub fn new() -> Self {
        Self {
            path: ActionsTaken::empty(),
            gain: 0.0,
        }
    }

    pub fn path(&self) -> &ActionsTaken {
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
