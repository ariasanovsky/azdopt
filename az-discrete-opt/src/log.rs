use core::mem::MaybeUninit;

use rayon::prelude::{
    IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
};

use crate::ir_min_tree::{ActionsTaken, Transitions};

pub struct BasicLog {
    pub(crate) path: ActionsTaken,
    pub(crate) gain: f32,
}

impl BasicLog {
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
        // let Self { path, gain } = self;
        // let Transitions {
        //     first_action,
        //     first_reward,
        //     transitions,
        //     end,
        // } = transitions;
        let end = transitions.end();
        let gain = end.gain();
        let gain_cmp = gain.total_cmp(&self.gain);
        let path = end.path();
        let length_cmp = path.len().cmp(&self.path.len());
        use core::cmp::Ordering;
        // prioritize gain, then length
        match (gain_cmp, length_cmp) {
            (Ordering::Greater, _) | (Ordering::Equal, Ordering::Greater) => {
                self.gain = gain;
                self.path = end.path().clone();
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
