use crate::nabla::space::NablaStateActionSpace;

pub struct StateNode {
    n_s: u32,
    c_s: f32,
    active_actions: Option<Vec<ActionData>>,
}

impl StateNode {
    pub fn next_action_data(&mut self) -> Option<(usize, &mut f32, &mut ActionDataKind)> {
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
            *a,
            g_theta_star_sa,
            kind,
        ))
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
    pub(super) fn new<Space: NablaStateActionSpace>(
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
    use super::*;

    #[test]
    fn test_nearest_integer() {
        let nearest_integers = (0..5).map(|j| rescaled_index(j, 11, 5)).collect::<Vec<_>>();
        assert_eq!(&nearest_integers, &[0, 3, 5, 8, 10]);
    }
}