use az_discrete_opt::state::State;

use crate::simple_graph::edge::Edge;

use super::BitsetGraph;

#[derive(Clone, PartialEq, Eq)]
pub enum Action {
    Add(Edge),
    Delete(Edge),
}

impl<const N: usize> az_discrete_opt::state::Action<BitsetGraph<N>> for Action {
    fn index(&self) -> usize {
        match self {
            Action::Add(e) => {
                let pos = e.colex_position();
                pos
            },
            Action::Delete(e) => {
                let pos = e.colex_position();
                pos + (N * (N - 1) / 2)
            },
        }
    }
}

impl<const N: usize> State for BitsetGraph<N> {
    type Actions = Action;

    fn actions(&self) -> impl Iterator<Item = Self::Actions> {
        let Self { neighborhoods } = self;
        neighborhoods.iter().enumerate().flat_map(|(v, n)| {
            (0..v).map(move |u| {
                let e = unsafe { Edge::new_unchecked(v, u) };
                if n.contains(u) {
                    Action::Delete(e)
                }  else {
                    Action::Add(e)
                }
            })
        })
    }

    unsafe fn act_unchecked(&mut self, action: &Self::Actions) {
        let Self { neighborhoods } = self;
        match action {
            Action::Add(e) | Action::Delete(e) => {
                let (v, u) = e.vertices();
                neighborhoods[u].add_or_remove_unchecked(v);
                neighborhoods[v].add_or_remove_unchecked(u);
            },
        }
    }
}

