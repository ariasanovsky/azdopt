use az_discrete_opt::{nabla::space::NablaStateActionSpace, space::axioms::{ActionsNeverRepeat, ActionOrderIndependent}};

use crate::{bitset::Bitset, simple_graph::edge::Edge};

use super::{ReassignColor, CountChange, TotalCounts, no_recolor::RamseyCountsNoRecolor};

pub struct RamseySpaceNoEdgeRecolor<B, const N: usize, const E: usize, const C: usize> {
    sizes: [usize; C],
    weights: [f32; C],
    _marker: core::marker::PhantomData<B>,
}

impl<B, const N: usize, const E: usize, const C: usize> RamseySpaceNoEdgeRecolor<B, N, E, C> {
    pub const fn new(
        sizes: [usize; C],
        weights: [f32; C],
    ) -> Self {
        Self {
            sizes,
            weights,
            _marker: core::marker::PhantomData,
        }
    }
}

impl<B, const N: usize, const E: usize, const C: usize> NablaStateActionSpace for RamseySpaceNoEdgeRecolor<B, N, E, C>
where
    B: Bitset + Clone,
    B::Bits: Clone,
{
    type State = RamseyCountsNoRecolor<N, E, C, B>;

    type Action = ReassignColor;

    type Reward = CountChange;

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

    fn reward(&self, state: &Self::State, index: usize) -> Self::Reward {
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

    fn action_data<'a>(&self, state: &'a Self::State) -> impl Iterator<Item = (usize, Self::Reward)> + 'a {
        let colors = (0..N).flat_map(move |v| (0..v).map(move |u| state.state.graph().color(v, u)));
        struct _ActionData {
            e_pos: usize,
            old_color: usize,
            new_color: usize,
        }
        let candidate_actions = colors.enumerate().flat_map(move |(e_pos, old_color)| {
            let new_colors = (0..old_color).chain(old_color+1..C);
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
                }
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
        let clique_counts = state.state.counts.iter().flat_map(|c| c.iter()).map(|c| *c as f32);
        let edge_bools = state.state.graph().graphs().iter().flat_map(|g| g.edge_bools()).map(|b| if b { 1.0f32 } else { 0. });
        let clique_edge = clique_counts.chain(edge_bools);
        clique_edge_vec.iter_mut().zip(clique_edge).for_each(|(v, c)| *v = c);
        for e_pos in state.permitted_edges.iter() {
            permit_vec[*e_pos] = 1.;
        }
    }

    fn cost(&self, state: &Self::State) -> Self::Cost {
        state.state.clique_counts().clone()
    }

    fn evaluate(&self, cost: &Self::Cost) -> f32 {
        cost.0.iter().zip(self.weights.iter()).map(|(c, w)| *c as f32 * w).sum()
    }

    fn g_theta_star_sa(&self, _c_s: &Self::Cost, r_sa: Self::Reward, h_theta_s_a: f32) -> f32 {
        let r_sa =
            r_sa.old_count as f32 * self.weights[r_sa.old_color] -
            r_sa.new_count as f32 * self.weights[r_sa.new_color]
            ;
        // r_sa + h_theta_s_a.powi(2)
        r_sa + h_theta_s_a
    }

    fn h_sa(&self, _c_s: f32, c_as: f32, c_as_star: f32) -> f32 {
        // (c_as - c_as_star).sqrt()
        c_as - c_as_star
    }
}

unsafe impl<B, const N: usize, const E: usize, const C: usize> ActionsNeverRepeat for RamseySpaceNoEdgeRecolor<B, N, E, C> {}
unsafe impl<B, const N: usize, const E: usize, const C: usize> ActionOrderIndependent for RamseySpaceNoEdgeRecolor<B, N, E, C> {}
