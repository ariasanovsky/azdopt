use crate::nabla::space::NablaStateActionSpace;

pub(super) struct StateNode {
    n_s: usize,
    c_s: f32,
    actions: Vec<ActionData>,
}

pub(super) struct ActionData {
    a: usize,
    n_sa: i32,
    g_theta_star_sa: f32,
}

pub(super) enum StateNodeKind {
    Active { node: StateNode },
    Exhausted { c_s_star: f32 },
}

impl StateNodeKind {
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
        }).collect();
        match actions.is_empty() {
            true => Self::Exhausted { c_s_star: c_s },
            false => {
                actions.sort_unstable_by(|a, b| b.g_theta_star_sa.partial_cmp(&a.g_theta_star_sa).unwrap());
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
                Self::Active {
                    node: StateNode {
                        n_s: 0,
                        c_s,
                        actions,
                    },
                }
            },
        }
    }
}

const fn rescaled_index(i: usize, l: usize, k: usize) -> usize {
    if k == 1 {
        panic!("k cannot be 1 because it would cause division by zero");
    }
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