use az_discrete_opt::state::State;

use crate::simple_graph::{bitset_graph::state::Action, edge::Edge};

use super::ConnectedBitsetGraph;

impl<const N: usize> az_discrete_opt::state::Action<ConnectedBitsetGraph<N>> for Action {
    fn index(&self) -> usize {
        <Self as az_discrete_opt::state::Action<crate::simple_graph::bitset_graph::BitsetGraph<N>>>::index(self)
    }
}

impl<const N: usize> State for ConnectedBitsetGraph<N> {
    type Actions = Action;

    fn actions(&self) -> impl Iterator<Item = Self::Actions> {
        let Self { neighborhoods } = self;
        neighborhoods.iter().enumerate().flat_map(move |(v, n)| {
            (0..v).filter_map(move |u| {
                let e = unsafe { Edge::new_unchecked(v, u) };
                if n.contains(u) {
                    if self.is_cut_edge(&e) {
                        Some(Action::Delete(e))
                    } else {
                        None
                    }
                }  else {
                    Some(Action::Add(e))
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

#[cfg(test)]
mod test {
    use crate::simple_graph::bitset_graph::BitsetGraph;

    #[test]
    fn c4_is_connected_and_has_no_cut_edges() {
        let graph: BitsetGraph<4> = [(0, 1), (1, 2), (2, 3), (3, 0)].as_ref().try_into().unwrap();
        let graph = graph.to_connected().unwrap();
        let len = graph.edges().enumerate().map(|(i, e)| {
            assert!(!graph.is_cut_edge(&e));
            i
        }).last().unwrap() + 1;
        assert_eq!(len, 4);
    }

    #[test]
    fn p4_is_connected_and_all_edges_are_cut_edges() {
        let graph: BitsetGraph<4> = [(0, 1), (1, 2), (2, 3)].as_ref().try_into().unwrap();
        let graph = graph.to_connected().unwrap();
        let len = graph.edges().enumerate().map(|(i, e)| {
            assert!(graph.is_cut_edge(&e));
            i
        }).last().unwrap() + 1;
        assert_eq!(len, 3);
    }

    #[test]
    fn paw_is_connected_and_has_one_cut_edge() {
        let graph: BitsetGraph<4> = [(0, 1), (0, 2), (1, 2), (2, 3)].as_ref().try_into().unwrap();
        let graph = graph.to_connected().unwrap();
        let len = graph.edges().enumerate().map(|(i, e)| {
            if graph.is_cut_edge(&e) {
                assert_eq!(e.vertices(), (3, 2));
            }
            i
        }).last().unwrap() + 1;
        assert_eq!(len, 4);
    }
}
