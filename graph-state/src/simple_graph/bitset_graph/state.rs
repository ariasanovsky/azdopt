use az_discrete_opt::state::State;

use crate::{simple_graph::edge::Edge, bitset::bitset::Bitset};

use super::BitsetGraph;

#[derive(Clone, PartialEq, Eq, Debug)]
pub enum Action {
    Add(Edge),
    Delete(Edge),
}

impl<const N: usize> az_discrete_opt::state::Action<BitsetGraph<N>> for Action {
    fn index(&self) -> usize {
        match self {
            Action::Add(e) => e.colex_position(),
            Action::Delete(e) => {
                let pos = e.colex_position();
                pos + (N * (N - 1) / 2)
            }
        }
    }

    unsafe fn from_index_unchecked(index: usize) -> Self {
        let e = N * (N - 1) / 2;
        if index < e {
            Self::Add(Edge::from_colex_position(index))
        } else {
            Self::Delete(Edge::from_colex_position(index - e))
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
                if n.contains(u as _).unwrap() {
                    Action::Delete(e)
                } else {
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
                neighborhoods[u].add_or_remove_unchecked(v as u32);
                neighborhoods[v].add_or_remove_unchecked(u as u32);
            }
        }
    }
}
