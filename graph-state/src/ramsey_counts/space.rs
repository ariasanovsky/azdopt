use az_discrete_opt::{nabla::space::NablaStateActionSpace, state::prohibit::WithProhibitions, space::axioms::{ActionsNeverRepeat, ActionOrderIndependent}};

use crate::{bitset::Bitset, simple_graph::edge::Edge};

use super::{RamseyCounts, ReassignColor, CountChange, TotalCounts};

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
    type State = WithProhibitions<RamseyCounts<N, E, C, B>>;

    type Action = ReassignColor;

    type Reward = CountChange;

    type Cost = TotalCounts<C>;

    const STATE_DIM: usize = E * C * 3;

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
        debug_assert!(!state.prohibited_actions.contains(&index));
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
        let WithProhibitions {
            state,
            prohibited_actions,
        } = state;
        
        let ReassignColor {
            edge_pos,
            new_color,
        } = *action;
        let edge = Edge::from_colex_position(edge_pos);
        state.reassign_color(edge, new_color, &self.sizes);
        prohibited_actions.extend((0..C).map(|c| edge_pos + E * c));
    }

    fn action_data<'a>(&self, state: &'a Self::State) -> impl Iterator<Item = (usize, Self::Reward)> + 'a {
        let colors = (0..N).flat_map(move |v| (0..v).map(move |u| state.state.graph().color(v, u)));
        struct _ActionData {
            i: usize,
            a: usize,
            old_color: usize,
            new_color: usize,
        }
        let candidate_actions = colors.enumerate().flat_map(move |(i, old_color)| {
            let new_colors = (0..old_color).chain(old_color+1..C);
            new_colors.map(move |new_color| _ActionData {
                i,
                a: i + E * new_color,
                old_color,
                new_color,
            })
        });
        candidate_actions.filter_map(move |a| match state.prohibited_actions.contains(&a.a) {
            true => None,
            false => Some((a.a, CountChange {
                old_color: a.old_color,
                new_color: a.new_color,
                old_count: state.state.counts[a.old_color][a.i],
                new_count: state.state.counts[a.new_color][a.i],
            })),
        })
    }

    fn write_vec(&self, state: &Self::State, vector: &mut [f32]) {
        debug_assert!(vector.len() == Self::STATE_DIM);
        vector.fill(0.);
        /* chunks are as follows:
        * 0/1: red/blue clique counts
        * 2/3: red/blue edge bools
        * 4/5: red/blue prohibited actions
        */
        let (clique_edge_vec, prohib_vec) = vector.split_at_mut(4 * E);
        let clique_counts = state.state.counts.iter().flat_map(|c| c.iter()).map(|c| *c as f32);
        let edge_bools = state.state.graph().graphs().iter().flat_map(|g| g.edge_bools()).map(|b| if b { 1.0f32 } else { 0. });
        let clique_edge = clique_counts.chain(edge_bools);
        clique_edge_vec.iter_mut().zip(clique_edge).for_each(|(v, c)| *v = c);
        for a in state.prohibited_actions.iter() {
            prohib_vec[*a] = 1.;
        }
    }

    fn cost(&self, state: &Self::State) -> Self::Cost {
        state.state.clique_counts().clone()
    }

    fn evaluate(&self, cost: &Self::Cost) -> f32 {
        cost.0.iter().zip(self.weights.iter()).map(|(c, w)| *c as f32 * w).sum()
    }

    // TODO: pass in c_s: f32 instead of &Self::Cost
    fn g_theta_star_sa(&self, c_s: &Self::Cost, r_sa: Self::Reward, h_theta_s_a: f32) -> f32 {
        // debug_assert!(h_theta_s_a >= 0.);
        let c_s = self.evaluate(c_s);
        let CountChange {
            old_color,
            new_color,
            old_count,
            new_count,
        } = r_sa;
        let reward =
            (old_count as f32 * self.weights[old_color]) - 
            (new_count as f32 * self.weights[new_color]);
        // (reward + h_theta_s_a).min(c_s)
        reward + h_theta_s_a * (c_s - reward)
    }

    // TODO: ?pass in c_s: f32 instead of &Self::Cost
    fn h_sa(&self, c_s: &Self::Cost, r_sa: Self::Reward, g_sa: f32) -> f32 {
        let c_s = self.evaluate(c_s);
        let CountChange {
            old_color,
            new_color,
            old_count,
            new_count,
        } = r_sa;
        let reward = 
            (old_count as f32 * self.weights[old_color]) - 
            (new_count as f32 * self.weights[new_color]);
        debug_assert!(g_sa >= reward);
        // (g_sa - reward).clamp(0., c_s - reward)
        (g_sa - reward) / (c_s - reward)
    }
}

unsafe impl<B, const N: usize, const E: usize, const C: usize> ActionsNeverRepeat for RamseySpaceNoEdgeRecolor<B, N, E, C> {}
unsafe impl<B, const N: usize, const E: usize, const C: usize> ActionOrderIndependent for RamseySpaceNoEdgeRecolor<B, N, E, C> {}
