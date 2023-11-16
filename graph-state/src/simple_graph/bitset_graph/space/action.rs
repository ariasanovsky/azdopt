use crate::simple_graph::edge::Edge;

#[derive(Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum AddOrDeleteEdge {
    Add(Edge),
    Delete(Edge),
}

impl AddOrDeleteEdge {
    pub fn action_index<const N: usize>(&self) -> usize {
        match self {
            Self::Add(e) => e.colex_position(),
            Self::Delete(e) => {
                let pos = e.colex_position();
                pos + N * (N - 1) / 2
            }
        }
    }

    pub fn from_action_index<const N: usize>(index: usize) -> Self {
        let e = N * (N - 1) / 2;
        if index < e {
            Self::Add(Edge::from_colex_position(index))
        } else {
            Self::Delete(Edge::from_colex_position(index - e))
        }
    }
}
