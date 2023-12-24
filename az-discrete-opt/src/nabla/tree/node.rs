use std::collections::VecDeque;

use crate::nabla::space::NablaStateActionSpace;

pub struct StateNode {
    n_s: u32,
    c_s: f32,
    active_actions: VecDeque<ActionData>,
    exhausted_actions: Vec<ActionData>,
}

impl StateNode {
    pub fn next_transition(&mut self) -> Option<Transition> {
        let Self {
            n_s,
            c_s,
            active_actions,
            exhausted_actions,
        } = self;
        if active_actions.is_empty() {
            return None;
        }
        if (*n_s as usize) < active_actions.len() {
            Some(Transition {
                c_s: *c_s,
                pos: *n_s,
                n_s,
                active_actions,
                exhausted_actions,
            })
        } else {
            todo!()
        }
        // let ActionData {
        //     a,
        //     n_sa,
        //     g_theta_star_sa,
        // } = if (*n_s as usize) < actions.len() {
        //     &mut actions[*n_s as usize]
        // } else {
        //     todo!()
        // };
        // *n_s += 1;
        // *n_sa += 1;
        // todo!()
    }

    pub fn cost(&self) -> f32 {
        self.c_s
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
    Exhausting,
    Active,
}

impl Transition<'_> {
    pub fn action_index(&self) -> usize {
        self.active_actions[self.pos as _].a
    }

    pub fn update_action_data(
        self,
        kind: &mut TransitionKind,
        c_theta_star: &mut f32,
    ) {
        let Self {
            n_s,
            c_s,
            pos,
            active_actions,
            exhausted_actions,
        } = self;
        match kind {
            TransitionKind::Exhausting => {
                let mut action = active_actions.remove(pos as _).unwrap();
                todo!("update the value inside action");
                exhausted_actions.push(action);
                match active_actions.is_empty() {
                    true => todo!(),
                    false => todo!(),
                }
            },
            TransitionKind::Active => {
                let action = active_actions.get_mut(pos as _).unwrap();
                action.update_with(c_s, c_theta_star);
                *n_s += 1;
            },
        }
    }
}

pub struct ActionData {
    pub(super) a: usize,
    pub(super) n_sa: u32,
    pub(super) g_theta_star_sa: f32,
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
        *g_theta_star_sa = g_sa.min(*g_theta_star_sa);
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
        max_num_actions: usize,
    ) -> Self {
        let c_s = space.evaluate(cost);
        let mut actions: VecDeque<ActionData> = space.action_data(&state).map(|(a, r_sa)| ActionData {
            a,
            n_sa: 0,
            g_theta_star_sa: space.g_theta_star_sa(c_s, r_sa, h_theta[a]),
        }).collect();
        actions.make_contiguous().sort_unstable_by(|a, b| a.g_theta_star_sa.partial_cmp(&b.g_theta_star_sa).unwrap());
        match actions.len() <= max_num_actions {
            true => {},
            false => {
                for j in 0..max_num_actions {
                    let i = rescaled_index(j, actions.len(), max_num_actions);
                    actions.swap(i, j);
                }
                actions.truncate(max_num_actions);
                actions.shrink_to_fit();
            },
        }
        Self {
            n_s: 0,
            c_s,
            active_actions: actions,
            exhausted_actions: Vec::new(),
        }
    }
}

const fn rescaled_index(i: usize, l: usize, k: usize) -> usize {
    (i * 2 * (l - 1) + k - 1) / ((k - 1) * 2)
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