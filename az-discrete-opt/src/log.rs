use core::mem::MaybeUninit;

use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator, IndexedParallelIterator, IntoParallelRefIterator};

use crate::ir_min_tree::{ActionsTaken, Transitions};

pub struct GraphLogs {
    pub(crate) path: ActionsTaken,
    pub(crate) gain: f32,
}

impl GraphLogs {
    pub fn empty() -> Self {
        Self {
            path: ActionsTaken::empty(),
            gain: 0.0,
        }
    }

    pub fn par_new_logs<const BATCH: usize>() -> [Self; BATCH] {
        let mut logs: [MaybeUninit<Self>; BATCH] = MaybeUninit::uninit_array();
        logs.par_iter_mut().for_each(|l| {
            l.write(Self::empty());
        });
        unsafe { MaybeUninit::array_assume_init(logs) }
    }

    pub fn update(&mut self, transitions: &Transitions) {
        let end = transitions.end();
        let gain = end.gain();
        match gain.total_cmp(&self.gain) {
            std::cmp::Ordering::Greater => {
                self.gain = gain;
                self.path = end.path().clone();
            }
            _ => {}
        }
    }
}

pub fn par_update_logs<const BATCH: usize>(logs: &mut [GraphLogs; BATCH], transitions: &[Transitions; BATCH]) {
    let logs = logs.par_iter_mut();
    let transitions = transitions.par_iter();
    logs.zip_eq(transitions).for_each(|(l, t)| {
        l.update(t)
    });
}

