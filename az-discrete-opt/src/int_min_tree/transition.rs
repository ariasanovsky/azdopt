use super::{StateDataKind, INTStateData};

#[derive(Debug)]
pub struct INTTransition<'a> {
    pub(crate) data_i: StateDataKindMutRef<'a>,
    pub(crate) kind: TransitionKind,
}

#[derive(Debug)]
pub enum StateDataKindMutRef<'a> {
    Exhausted { c_t_star: f32 },
    Active { data: &'a mut INTStateData },
}


impl<'a> INTTransition<'a> {
    pub fn index(&self) -> usize {
        match self {
            Self { data_i: StateDataKindMutRef::Exhausted { c_t_star: _ }, kind: _ } => unreachable!("Exhausted state has no best action"),
            Self { data_i: StateDataKindMutRef::Active { data }, kind } => match kind {
                TransitionKind::LastUnvisitedAction => data.unvisited_actions.last().unwrap().a,
                TransitionKind::LastVisitedAction => data.visited_actions.last().unwrap().a,
            },
        }
    }

    pub fn update(&self, c_star_theta_i: f32) {
        todo!("refactor s.t. `c_star_theta_i` carries the information of whether we are exhausting");
        todo!("if exhausting, `data_i` is necessarily `Active`");
        match self.kind {
            TransitionKind::LastUnvisitedAction => {
                todo!("remove the last unvisited action");
                todo!("move it into the visited actions");
                todo!()
            },
            TransitionKind::LastVisitedAction => {
                todo!()
            },
        }
        todo!("sort the visited actions")
    }

    pub fn c_i_star(&self) -> f32 {
        match &self.data_i {
            StateDataKindMutRef::Exhausted { c_t_star } => *c_t_star,
            StateDataKindMutRef::Active { data } => data.c_star,
        }
    }
}

#[derive(Debug)]
pub enum TransitionKind {
    LastUnvisitedAction,
    LastVisitedAction,
}