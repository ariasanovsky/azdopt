use az_discrete_opt::{
    nabla::space::NablaStateActionSpace,
    space::axioms::{ActionOrderIndependent, ActionsNeverRepeat},
};

use crate::{bitset::Bitset, simple_graph::edge::Edge};

use super::{no_recolor::RamseyCountsNoRecolor, CountChange, ReassignColor, TotalCounts};

pub struct RamseySpaceNoEdgeRecolor<B, const N: usize, const E: usize, const C: usize> {
    sizes: [usize; C],
    weights: [f32; C],
    _marker: core::marker::PhantomData<B>,
}

impl<B, const N: usize, const E: usize, const C: usize> RamseySpaceNoEdgeRecolor<B, N, E, C> {
    pub const fn new(sizes: [usize; C], weights: [f32; C]) -> Self {
        Self {
            sizes,
            weights,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<B, const N: usize, const E: usize, const C: usize> NablaStateActionSpace
    for RamseySpaceNoEdgeRecolor<B, N, E, C>
where
    B: Bitset + Clone,
    B::Bits: Clone,
{
    type State = RamseyCountsNoRecolor<N, E, C, B>;

    type Action = ReassignColor;

    type RewardHint = CountChange;

    type Cost = TotalCounts<C>;

    const STATE_DIM: usize = E * (2 * C + 1);

    const ACTION_DIM: usize = E * C;

    // fn index(&self, action: &Self::Action) -> usize {
    //     todo!()
    // }

    fn action(&self, index: usize) -> Self::Action {
        debug_assert!(index < Self::ACTION_DIM);
        ReassignColor {
            edge_pos: index % E,
            new_color: index / E,
        }
    }

    fn reward(&self, state: &Self::State, index: usize) -> Self::RewardHint {
        debug_assert!(index < Self::ACTION_DIM);
        debug_assert!(state.permitted_edges.contains(&(index % E)));
        let new_color = index / E;
        let index = index % E;
        let edge = Edge::from_colex_position(index);
        let (v, u) = edge.vertices();
        let old_color = state.state.graph().color(v, u);
        let old_count = state.state.counts[old_color][index];
        let new_count = state.state.counts[new_color][index];
        CountChange {
            old_color,
            new_color,
            old_count,
            new_count,
        }
    }

    fn act(&self, state: &mut Self::State, action: &Self::Action) {
        let RamseyCountsNoRecolor {
            state,
            permitted_edges,
        } = state;

        let ReassignColor {
            edge_pos,
            new_color,
        } = *action;
        let edge = Edge::from_colex_position(edge_pos);
        state.reassign_color(edge, new_color, &self.sizes);
        let removed = permitted_edges.remove(&edge_pos);
        debug_assert!(removed);
    }

    fn action_data<'a>(
        &self,
        state: &'a Self::State,
    ) -> impl Iterator<Item = (usize, Self::RewardHint)> + 'a {
        let colors = (0..N).flat_map(move |v| (0..v).map(move |u| state.state.graph().color(v, u)));
        struct _ActionData {
            e_pos: usize,
            old_color: usize,
            new_color: usize,
        }
        let candidate_actions = colors.enumerate().flat_map(move |(e_pos, old_color)| {
            let new_colors = (0..old_color).chain(old_color + 1..C);
            new_colors.map(move |new_color| _ActionData {
                e_pos,
                old_color,
                new_color,
            })
        });
        candidate_actions.filter_map(move |a| match state.permitted_edges.contains(&a.e_pos) {
            false => None,
            true => Some((
                a.e_pos + a.new_color * E,
                CountChange {
                    old_color: a.old_color,
                    new_color: a.new_color,
                    old_count: state.state.counts[a.old_color][a.e_pos],
                    new_count: state.state.counts[a.new_color][a.e_pos],
                },
            )),
        })
    }

    fn write_vec(&self, state: &Self::State, vector: &mut [f32]) {
        debug_assert!(vector.len() == Self::STATE_DIM);
        vector.fill(0.);
        /* chunks are as follows:
         * 0..(E * C): clique counts
         * (E * C)..(2 * E * C): edge bools
         * (2 * E * C)..(2 * E * C + E): permitted edges
         */
        let (clique_edge_vec, permit_vec) = vector.split_at_mut(2 * C * E);
        let clique_counts = state
            .state
            .counts
            .iter()
            .flat_map(|c| c.iter())
            .map(|c| *c as f32);
        let edge_bools = state
            .state
            .graph()
            .graphs()
            .iter()
            .flat_map(|g| g.edge_bools())
            .map(|b| if b { 1.0f32 } else { 0. });
        let clique_edge = clique_counts.chain(edge_bools);
        clique_edge_vec
            .iter_mut()
            .zip(clique_edge)
            .for_each(|(v, c)| *v = c);
        for e_pos in state.permitted_edges.iter() {
            permit_vec[*e_pos] = 1.;
        }
    }

    fn cost(&self, state: &Self::State) -> Self::Cost {
        state.state.clique_counts().clone()
    }

    fn evaluate(&self, cost: &Self::Cost) -> f32 {
        cost.0
            .iter()
            .zip(self.weights.iter())
            .map(|(c, w)| *c as f32 * w)
            .sum()
    }

    fn g_theta_star_sa(&self, c_s: f32, r_sa: Self::RewardHint, h_theta_sa: f32) -> f32 {
        let r_sa = r_sa.old_count as f32 * self.weights[r_sa.old_color]
            - r_sa.new_count as f32 * self.weights[r_sa.new_color];
        // let c_as = c_s - r_sa;
        // h_theta_sa * c_as + r_sa
        r_sa.max(c_s - 40. * h_theta_sa)
    }

    fn h_sa(&self, _c_s: f32, c_as: f32, c_as_star: f32) -> f32 {
        // c_as - c_as_star
        // 1. - c_as_star / c_as
        c_as_star / 40.
    }
}

unsafe impl<B, const N: usize, const E: usize, const C: usize> ActionsNeverRepeat
    for RamseySpaceNoEdgeRecolor<B, N, E, C>
{
}
unsafe impl<B, const N: usize, const E: usize, const C: usize> ActionOrderIndependent
    for RamseySpaceNoEdgeRecolor<B, N, E, C>
{
}

#[cfg(test)]
mod tests {
    use az_discrete_opt::nabla::space::NablaStateActionSpace;
    use rand::seq::SliceRandom;
    use rand_distr::WeightedIndex;

    use crate::{
        bitset::primitive::B32,
        ramsey_counts::{
            no_recolor::RamseyCountsNoRecolor, space::RamseySpaceNoEdgeRecolor, RamseyCounts,
        },
        simple_graph::{
            bitset_graph::{BitsetGraph, ColoredCompleteBitsetGraph},
            edge::Edge,
        },
    };

    #[test]
    fn correct_triangle_counts_after_modifying_a_random_3_edge_colored_graph_on_30_vertices() {
        const N: usize = 30;
        const E: usize = N * (N - 1) / 2;
        const C: usize = 3;
        type B = B32;
        type Space = RamseySpaceNoEdgeRecolor<B, N, E, C>;
        type State = RamseyCountsNoRecolor<N, E, C, B>;
        type Counts = RamseyCounts<N, E, C, B>;
        const SIZES: [usize; C] = [3, 3, 3];
        const WEIGHTS: [f32; C] = [1., 1., 1.];
        let space = Space::new(SIZES, WEIGHTS);
        let dist = WeightedIndex::new([1., 1., 1.]).unwrap();
        let mut rng = rand::thread_rng();
        let counts = Counts::generate(&mut rng, &dist, &SIZES);
        let mut state = State::generate(&mut rng, counts, E);
        while !space.is_terminal(&state) {
            let actions = space.action_data(&state).collect::<Vec<_>>();
            let (action_pos, _count_change) = actions.choose(&mut rng).unwrap().clone();
            let action = space.action(action_pos);
            let mut next_state = state.clone();
            space.act(&mut next_state, &action);
            let next_graph = next_state.state.graph().clone();
            let next_counts_recalculated = Counts::new(next_graph, &SIZES);
            for c in 0..C {
                let edges = Edge::edges::<N>();
                for (i, e) in edges.enumerate() {
                    assert_eq!(
                        next_counts_recalculated.counts[c][i], next_state.state.counts[c][i],
                        "
action: {action_pos}, {action:?}
edge: {e:?}
graph:\n{}
",
                        &state.state,
                    );
                }
            }
            space.act(&mut state, &action);
        }
    }

    #[test]
    fn correct_k4_counts_after_modifying_a_random_2_edge_colored_graph_on_15_vertices() {
        const N: usize = 5;
        const E: usize = N * (N - 1) / 2;
        const C: usize = 2;
        type B = B32;
        type Space = RamseySpaceNoEdgeRecolor<B, N, E, C>;
        type State = RamseyCountsNoRecolor<N, E, C, B>;
        type Counts = RamseyCounts<N, E, C, B>;
        const SIZES: [usize; C] = [4, 4];
        const WEIGHTS: [f32; C] = [1., 1.];
        let space = Space::new(SIZES, WEIGHTS);
        let dist = WeightedIndex::new([1., 1.]).unwrap();
        let mut rng = rand::thread_rng();
        let counts = Counts::generate(&mut rng, &dist, &SIZES);
        let mut state = State::generate(&mut rng, counts, E);
        while !space.is_terminal(&state) {
            let actions = space.action_data(&state).collect::<Vec<_>>();
            let (action_pos, _count_change) = actions.choose(&mut rng).unwrap().clone();
            let action = space.action(action_pos);
            let mut next_state = state.clone();
            space.act(&mut next_state, &action);
            let next_graph = next_state.state.graph().clone();
            let next_counts_recalculated = Counts::new(next_graph, &SIZES);
            for c in 0..C {
                let edges = Edge::edges::<N>();
                for (i, e) in edges.enumerate() {
                    assert_eq!(
                        next_counts_recalculated.counts[c][i], next_state.state.counts[c][i],
                        "
action: {action_pos}, {action:?}
edge: {e:?}
graph:\n{}
counts:\n{:?}
",
                        &state.state, &next_state.state.counts
                    );
                }
            }
            space.act(&mut state, &action);
        }
    }

    #[test]
    fn making_edge_0_1_red_in_a_specific_2_edge_colored_k5_produces_the_correct_clique_counts() {
        let red_edges = vec![(0, 2), (0, 3), (1, 2), (1, 3), (2, 3)];
        let blue_edges = vec![(0, 1), (0, 4), (1, 4), (2, 4), (3, 4)];
        type B = B32;
        const N: usize = 5;
        let red_graph = BitsetGraph::<N, B>::try_from(&red_edges[..]).unwrap();
        let blue_graph = BitsetGraph::<N, B>::try_from(&blue_edges[..]).unwrap();
        let colored_graph = ColoredCompleteBitsetGraph {
            graphs: [red_graph, blue_graph],
        };
        const C: usize = 2;
        const SIZES: [usize; C] = [4, 4];
        const E: usize = N * (N - 1) / 2;
        let counts = RamseyCounts::<N, E, C, _>::new(colored_graph, &SIZES);
        assert_eq!(
            &counts.counts,
            &[
                [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            ]
        );
        // todo!()
    }
}
