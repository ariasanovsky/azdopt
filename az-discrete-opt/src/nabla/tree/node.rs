use std::collections::VecDeque;

use crate::nabla::space::NablaStateActionSpace;

pub struct StateNode {
    n_s: u32,
    c_s: f32,
    active_actions: VecDeque<ActionData>,
    exhausted_actions: Vec<ActionData>,
}

pub enum SearchPolicy {
    Cyclic,
    Rating(fn(TransitionMetadata) -> f32),
}

pub struct TransitionMetadata {
    pub n_s: u32,
    pub c_s: f32,
    pub n_sa: u32,
    pub g_theta_star_sa: f32,
}

impl StateNode {
    pub fn next_transition(
        &mut self,
        policy: SearchPolicy,
    ) -> Result<Transition, f32> {
        let Self {
            n_s,
            c_s,
            active_actions,
            exhausted_actions,
        } = self;
        if active_actions.is_empty() {
            let c_s_theta_star =
                exhausted_actions.iter()
                .map(|a| a.g_theta_star_sa)
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .map_or(*c_s, |g_theta_star_sa| *c_s - g_theta_star_sa);
            return Err(c_s_theta_star)
        }
        Ok(match policy {
            SearchPolicy::Cyclic => Transition {
                c_s: *c_s,
                pos: *n_s % (active_actions.len() as u32),
                n_s,
                active_actions,
                exhausted_actions,
            },
            SearchPolicy::Rating(rating) => {
                let pos = active_actions.iter().enumerate().map(|(i, a)| {
                    let metadata = TransitionMetadata {
                        n_s: *n_s,
                        c_s: *c_s,
                        n_sa: a.n_sa,
                        g_theta_star_sa: a.g_theta_star_sa,
                    };
                    (i, rating(metadata))
                }).max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).map(|(i, _)| i).unwrap();
                Transition {
                    n_s,
                    c_s: *c_s,
                    pos: pos as _,
                    active_actions,
                    exhausted_actions,
                }
            },
        })
        // if (*n_s as usize) < active_actions.len() {
        //     Ok(Transition {
        //         c_s: *c_s,
        //         pos: *n_s,
        //         n_s,
        //         active_actions,
        //         exhausted_actions,
        //     })
        // } else {
        //     // todo! temporary cyclic policy
        //     let pos = *n_s % (active_actions.len() as u32);
        //     Ok(Transition {
        //         c_s: *c_s,
        //         pos,
        //         n_s,
        //         active_actions,
        //         exhausted_actions,
        //     })
        // }
    }

    pub fn cost(&self) -> f32 {
        self.c_s
    }

    pub fn active_actions(&self) -> impl Iterator<Item = &ActionData> {
        self.active_actions.iter()
    }

    pub fn exhausted_actions(&self) -> impl Iterator<Item = &ActionData> {
        self.exhausted_actions.iter()
    }

    #[cfg(feature = "rayon")]
    pub(crate) fn _par_next_roots(&self) -> impl rayon::iter::ParallelIterator<Item = (usize, f32)> + '_ {
        use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

        let actions = self.active_actions.par_iter().chain(self.exhausted_actions.par_iter());
        actions.map(|a| {
            let ActionData { a, n_sa: _, g_theta_star_sa } = a;
            (
                *a,
                self.c_s - *g_theta_star_sa,
            )
        })
    }
}

pub struct Transition<'roll_out> {
    n_s: &'roll_out mut u32,
    c_s: f32,
    pos: u32,
    active_actions: &'roll_out mut VecDeque<ActionData>,
    exhausted_actions: &'roll_out mut Vec<ActionData>,
}

pub enum TransitionKind {
    Exhausting { c_theta_star: f32 },
    Active { c_theta_star: f32 },
}

impl Transition<'_> {
    pub fn action_index(&self) -> usize {
        self.active_actions[self.pos as _].a
    }

    pub fn update_action_data(
        self,
        kind: &mut TransitionKind,
    ) {
        let Self {
            n_s,
            c_s,
            pos,
            active_actions,
            exhausted_actions,
        } = self;
        match kind {
            TransitionKind::Exhausting { c_theta_star } => {
                let mut action = active_actions.remove(pos as _).unwrap();
                action.update_with(c_s, c_theta_star);
                exhausted_actions.push(action);
                match active_actions.is_empty() {
                    true => {},
                    false => *kind = TransitionKind::Active { c_theta_star: *c_theta_star },
                }
            },
            TransitionKind::Active { c_theta_star } => {
                let action = active_actions.get_mut(pos as _).unwrap();
                action.update_with(c_s, c_theta_star);
                *n_s += 1;
            },
        }
    }
}

#[derive(Debug)]
pub struct ActionData {
    pub a: usize,
    pub n_sa: u32,
    pub g_theta_star_sa: f32,
}

impl ActionData {
    fn update_with(
        &mut self,
        c_s: f32,
        c_theta_star: &mut f32,
    ) {
        let Self {
            a: _,
            n_sa,
            g_theta_star_sa,
        } = self;
        let g_sa = c_s - *c_theta_star;
        if *n_sa == 0 {
            *g_theta_star_sa = g_sa;
        } else {
            *g_theta_star_sa = g_sa.max(*g_theta_star_sa);
        }
        *c_theta_star = c_theta_star.min(c_s);
        *n_sa += 1;
    }
}

impl StateNode {
    pub fn new<Space: NablaStateActionSpace>(
        space: &Space,
        state: &Space::State,
        cost: &Space::Cost,
        h_theta: &[f32],
        action_sample_pattern: SamplePattern,
    ) -> (Self, TransitionKind) {
        let c_s = space.evaluate(cost);
        let mut actions: VecDeque<ActionData> = space.action_data(state).map(|(a, r_sa)| ActionData {
            a,
            n_sa: 0,
            g_theta_star_sa: space.g_theta_star_sa(cost, r_sa, h_theta[a]),
        }).collect();
        actions.make_contiguous().sort_unstable_by(|a, b| b.g_theta_star_sa.partial_cmp(&a.g_theta_star_sa).unwrap());
        if sample_swap(actions.make_contiguous(), &action_sample_pattern) {
            actions.truncate(action_sample_pattern.len());
            actions.shrink_to_fit();
        }
        let kind = match actions.front() {
            Some(a_star) => TransitionKind::Active { c_theta_star: c_s - a_star.g_theta_star_sa },
            None => TransitionKind::Exhausting { c_theta_star: c_s },
        };
        (
            Self {
                n_s: 0,
                c_s,
                active_actions: actions,
                exhausted_actions: Vec::new(),
            },
            kind,
        )
    }
}

const fn rescaled_index(i: usize, l: usize, k: usize) -> usize {
    (i * 2 * (l - 1) + k - 1) / ((k - 1) * 2)
}

/// A pattern for selecting representatives from a set of samples.
/// Retains:
/// - `head` elements from the head of the slice,
/// - `body` elements uniformly from the body of the slice, and
/// - `tail` elements from the tail of the slice.
#[derive(Clone)]
pub struct SamplePattern {
    pub head: usize,
    pub body: usize,
    pub tail: usize,
}

impl SamplePattern {
    pub fn len(&self) -> usize {
        self.head + self.body + self.tail
    }
}


pub(crate) fn sample_swap<T>(slice: &mut [T], pattern: &SamplePattern) -> bool {
    let SamplePattern { head, body, tail } = *pattern;
    if slice.len() < head + body + tail {
        return false
    }
    let (_, slice) = slice.split_at_mut(head);
    let end_of_body = slice.len() - tail;
    let (middle, _) = slice.split_at_mut(end_of_body);
    if body == 1 {
        let i = middle.len() / 2;
        middle.swap(0, i);
    } else {
        for j in 0..body {
            let i = rescaled_index(j, middle.len(), body);
            middle.swap(i, j);
        }
    }
    let (_, end) = slice.split_at_mut(body);
    let old_end = end.len() - tail;
    for j in 0..tail {
        end.swap(j, old_end + j);
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_nearest_integer() {
        let nearest_integers = (0..5).map(|j| rescaled_index(j, 11, 5)).collect::<Vec<_>>();
        assert_eq!(&nearest_integers, &[0, 3, 5, 8, 10]);
    }
}