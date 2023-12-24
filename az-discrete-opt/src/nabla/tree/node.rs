use crate::nabla::space::NablaStateActionSpace;

pub struct StateNode {
    n_s: u32,
    c_s: f32,
    active_actions: Option<Vec<ActionData>>,
}

impl StateNode {
    pub fn next_action_data(&mut self) -> Option<(f32, usize, &mut f32, &mut ActionDataKind)> {
        let Self {
            n_s,
            c_s: _,
            active_actions,
         } = self;
        let actions = active_actions.as_mut()?;
        
        let ActionData {
            a,
            n_sa,
            g_theta_star_sa,
            kind
        } = if (*n_s as usize) < actions.len() {
            &mut actions[*n_s as usize]
        } else {
            todo!()
        };
        *n_s += 1;
        *n_sa += 1;
        Some((
            self.c_s,
            *a,
            g_theta_star_sa,
            kind,
        ))
    }

    pub fn cost(&self) -> f32 {
        self.c_s
    }
}

pub struct ActionData {
    pub(super) a: usize,
    pub(super) n_sa: u32,
    pub(super) g_theta_star_sa: f32,
    pub(super) kind: ActionDataKind,
}

pub enum ActionDataKind {
    Active,
    Terminal,
    Exhausted,
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
        let mut actions: Vec<ActionData> = space.action_data(&state).map(|(a, r_sa)| ActionData {
            a,
            n_sa: 0,
            g_theta_star_sa: space.g_theta_star_sa(c_s, r_sa, h_theta[a]),
            kind: ActionDataKind::Active,
        }).collect();
        match actions.is_empty() {
            true => Self {
                n_s: 0,
                c_s,
                active_actions: None,
            },
            false => {
                actions.sort_unstable_by(|a, b| a.g_theta_star_sa.partial_cmp(&b.g_theta_star_sa).unwrap());
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
                    active_actions: Some(actions),
                }
            },
        }
    }
}

const fn rescaled_index(i: usize, l: usize, k: usize) -> usize {
    (i * 2 * (l - 1) + k - 1) / ((k - 1) * 2)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;

    #[test]
    fn test_nearest_integer() {
        let nearest_integers = (0..5).map(|j| rescaled_index(j, 11, 5)).collect::<Vec<_>>();
        assert_eq!(&nearest_integers, &[0, 3, 5, 8, 10]);
    }

    #[test]
    fn foo() {
        let mut src = VecDeque::new();
        src.push_back(0);
        src.push_back(1);
        src.push_back(2);
        src.push_back(3);
        src.push_back(4);

        // let x = buf.get_mut(2);
        // dbg!(x);

        struct DelayedMove<'a, T> {
            src: &'a mut VecDeque<T>,
            dst: &'a mut VecDeque<T>,
            i: usize,
        }

        impl<'a, T> DelayedMove<'a, T> {
            fn new(src: &'a mut VecDeque<T>, dst: &'a mut VecDeque<T>, i: usize) -> Option<Self> {
                if i < src.len() {
                    Some(Self { src, dst, i })
                } else {
                    None
                }
            }

            fn move_to_dst(self) {
                let DelayedMove { src, dst, i } = self;
                let x = src.remove(i).unwrap();
                dst.push_back(x);
            }
        }
        let mut dst = VecDeque::new();
        dbg!(&src, &dst);
        let delayed_move = DelayedMove::new(&mut src, &mut dst, 2).unwrap();
        delayed_move.move_to_dst();
        dbg!(&src, &dst);
    }
}