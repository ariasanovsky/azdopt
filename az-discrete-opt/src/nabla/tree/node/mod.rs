use crate::nabla::space::NablaStateActionSpace;

use self::action::ActionData;

pub mod action;

pub struct StateNode {
    pub(crate) c: f32,
    pub(crate) c_star: f32,
    pub(crate) actions: Vec<ActionData>,
}

impl StateNode {
    pub(crate) fn new<Space: NablaStateActionSpace>(
        space: &Space,
        state: &Space::State,
        cost: &Space::Cost,
        h_theta: &[f32],
    ) -> Self {
        let c = space.evaluate(cost);
        Self {
            c,
            c_star: c,
            actions: space
                .action_data(state)
                .map(|(a, r)| {
                    let g_sa = space.g_theta_star_sa(cost, r, h_theta[a]);
                    ActionData::new_predicted(a, g_sa)
                })
                .collect(),
        }
    }

    pub(crate) fn next_action(&mut self) -> Option<(usize, &mut ActionData)> {
        self.actions
            .iter_mut()
            .enumerate()
            .filter_map(|(i, a)| a.g_sa().map(|g_sa| (i, a, g_sa)))
            .max_by(|(_, _, a), (_, _, b)| a.partial_cmp(b).unwrap())
            .map(|(i, a, _)| (i, a))
    }

    pub(crate) fn is_exhausted(&self) -> bool {
        self.actions.iter().all(|a| a.g_sa().is_none())
    }

    pub(crate) fn new_exhausted(c: f32) -> Self {
        Self {
            c,
            c_star: c,
            actions: Default::default(),
        }
    }

    // pub(crate) fn update_c_star(&mut self, c_star: f32, action_position: usize, decay: f32) {
    //     debug_assert!(self.actions[action_position].g_sa().is_some());
    //     if self.c_star > c_star {
    //         // println!("improve node c_star!");
    //         self.c_star = c_star;
    //         let g = self.c - c_star;
    //         debug_assert!(g > 0.0);
    //         self.actions[action_position].update_g_sa(g);
    //     } else {
    //         self.actions[action_position].decay(decay);
    //     }
    // }

    pub(crate) fn update_c_star_and_decay(
        &mut self,
        action_position: usize,
        parent_c_star: f32,
        decay: f32,
    ) {
        let action_data = &mut self.actions[action_position];
        let g_sa = self.c - parent_c_star;
        if self.c_star > parent_c_star {
            self.c_star = parent_c_star;
            action_data.update_g_sa(g_sa);
        } else {
            action_data.decay(decay, g_sa);
        }
    }

    pub(crate) fn update_c_star_and_exhaust(
        &mut self,
        action_position: usize,
        parent_c_star: f32,
    ) {
        self.c_star = self.c_star.min(parent_c_star);
        let action_data = &mut self.actions[action_position];
        action_data.exhaust();
    }

    // pub(crate) fn exhaust_action(&mut self, action_position: usize) {
    //     self.actions[action_position].exhaust();
    // }
}
