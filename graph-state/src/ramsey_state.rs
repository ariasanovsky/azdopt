use core::mem::{transmute, MaybeUninit};

use az_discrete_opt::iq_min_tree::IQState;
use bit_iter::BitIter;
use itertools::Itertools;
use priority_queue::PriorityQueue;
// use rayon::prelude::{
//     IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator,
// };

use crate::{
    CliqueCounts, Color, ColoredCompleteGraph, EdgeRecoloring, MulticoloredGraphEdges,
    MulticoloredGraphNeighborhoods, OrderedEdgeRecolorings, C, E, N,
};

/* todo! remove globals
  * macros (and traits) to handle:
    * E-sized arrays (no const generics ;____;)
    * optimal choices of bitsets
  * generalize for clique size != 4
  * generalize for imbalanced weights for different clique sizes
    * renormalize w.r.t. the expected counts in the optimal G(n, \vec{p}) graph
      * solve for optimum w/ lagrange multiplier
*/
const ACTION: usize = C * E;

pub const STATE: usize = 6 * E + 1;
pub type StateVec = [f32; STATE];
pub type ActionVec = [f32; ACTION];
pub type ValueVec = [f32; 1];

pub trait Cost {
    fn cost(&self) -> f32;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RamseyState {
    colors: ColoredCompleteGraph,
    edges: MulticoloredGraphEdges,
    neighborhoods: MulticoloredGraphNeighborhoods,
    available_actions: [[bool; E]; C],
    ordered_actions: OrderedEdgeRecolorings,
    counts: CliqueCounts,
    time_remaining: usize,
}

impl RamseyState {
    pub fn all_red() -> Self {
        let colors: [Color; E] = core::array::from_fn(|_| Color(0));
        let mut edges: [[bool; E]; C] = [[false; E]; C];
        edges[0].iter_mut().for_each(|b| *b = true);
        let mut neighborhoods: [[u32; N]; C] = [[0; N]; C];
        neighborhoods[0]
            .iter_mut()
            .enumerate()
            .for_each(|(i, neigh)| {
                *neigh = (1 << N) - 1;
                *neigh ^= 1 << i;
            });
        let mut available_actions: [[bool; E]; C] = [[true; E]; C];
        available_actions[0].iter_mut().for_each(|b| *b = false);
        let time_remaining = 1;
        Self::new(
            colors,
            edges,
            neighborhoods,
            available_actions,
            time_remaining,
        )
    }

    fn new(
        colors: [Color; E],
        edges: [[bool; E]; C],
        neighborhoods: [[u32; N]; C],
        available_actions: [[bool; E]; C],
        time_remaining: usize,
    ) -> Self {
        let mut counts: [[MaybeUninit<i32>; E]; C] = unsafe { MaybeUninit::uninit().assume_init() };
        neighborhoods
            .iter()
            .zip(counts.iter_mut())
            .for_each(|(neighborhoods, counts)| {
                let edge_iterator = (0..N).flat_map(|v| (0..v).map(move |u| (u, v)));
                edge_iterator
                    .zip(counts.iter_mut())
                    .for_each(|((u, v), k)| {
                        let neighborhood = neighborhoods[u] & neighborhoods[v];
                        let count = BitIter::from(neighborhood)
                            .map(|w| (neighborhood & neighborhoods[w]).count_ones())
                            .sum::<u32>()
                            / 2;
                        k.write(count as i32);
                    });
            });
        let counts: [[i32; E]; C] = unsafe { transmute(counts) };
        let mut recolorings: PriorityQueue<EdgeRecoloring, i32> = PriorityQueue::new();
        colors.iter().enumerate().for_each(|(i, c)| {
            let old_color = c.0;
            let old_count = counts[old_color][i];
            (0..C).filter(|c| old_color.ne(c)).for_each(|new_color| {
                let new_count = counts[new_color][i];
                let reward = old_count - new_count;
                let recoloring = EdgeRecoloring {
                    new_color,
                    edge_position: i,
                };
                recolorings.push(recoloring, reward);
            })
        });

        // (0..C).for_each(|c| {
        //     (0..N).for_each(|u| {
        //         let neigh = neighborhoods[c][u];
        //         assert_eq!(neigh & (1 << u), 0, "u = {u}, neigh = {neigh:b}");
        //     });
        // });

        // s.check_for_inconsistencies();
        // s
        Self {
            colors: ColoredCompleteGraph(colors),
            edges: MulticoloredGraphEdges(edges),
            neighborhoods: MulticoloredGraphNeighborhoods(neighborhoods),
            available_actions,
            ordered_actions: OrderedEdgeRecolorings(recolorings),
            counts: CliqueCounts(counts),
            time_remaining,
        }
    }

    pub fn to_vec(&self) -> StateVec {
        let Self {
            colors: _,
            edges: MulticoloredGraphEdges(edges),
            neighborhoods: _,
            available_actions,
            ordered_actions: _,
            counts: CliqueCounts(counts),
            time_remaining,
        } = self;
        let edge_iter = edges.iter().flatten().map(|b| if *b { 1.0 } else { 0.0 });
        let count_iter = counts.iter().flatten().map(|c| *c as f32);
        let action_iter = available_actions
            .iter()
            .flatten()
            .map(|a| if *a { 1.0 } else { 0.0 });
        let time_iter = Some(*time_remaining as f32).into_iter();
        let state_iter = edge_iter
            .chain(count_iter)
            .chain(action_iter)
            .chain(time_iter);
        let mut state_vec: StateVec = [0.0; STATE];
        state_vec.iter_mut().zip(state_iter).for_each(|(v, s)| {
            *v = s;
        });
        state_vec
    }
}

/* edges are enumerated in colex order:
    [0] 01  [1] 02  [3] 03
            [2] 12  [4] 13
                    [5] 23
*/
pub fn edge_from_position(position: usize) -> (usize, usize) {
    /* note the positions
        {0, 1}: 0 = (2 choose 2) - 1
        {1, 2}: 2 = (3 choose 2) - 1    increased by 2
        {2, 3}: 5 = (4 choose 2) - 1    increased by 3
        {3, 4}: 9 = (5 choose 2) - 1    increased by 4
        ...
        {v-1, v}: (v+1 choose 2) - 1
    */
    // the smart thing is a lookup table or direct computation
    /* solve for v from position
         (v+1 choose 2) - 1 = position
         8 * (v+1 choose 2) = 8 * position + 8
         (2*v + 1)^2 = 8 * position + 9
            2*v + 1 = sqrt(8 * position + 9)
            v = (sqrt(8 * position + 9) - 1) / 2
        etc
    */
    // todo!() we do a lazy linear search
    let mut v = 1;
    let mut upper_position = 0;
    while upper_position < position {
        v += 1;
        upper_position += v;
    }
    let difference = upper_position - position;
    let u = v - difference - 1;
    debug_assert_ne!(u, v, "{position} -> {upper_position} -> {difference}");
    (u, v)
}

pub fn edge_to_position(u: usize, v: usize) -> usize {
    let (u, v) = if u < v { (u, v) } else { (v, u) };
    /* note the positions
        {0, 1}: 0 = (2 choose 2) - 1
        {1, 2}: 2 = (3 choose 2) - 1
        {2, 3}: 5 = (4 choose 2) - 1
        {3, 4}: 9 = (5 choose 2) - 1
        ...
        {v-1, v}: (v+1 choose 2) - 1
    */
    let upper_position = v * (v + 1) / 2;
    // subtract the difference between u & v
    let difference = v - u;
    upper_position - difference
}

impl IQState<STATE> for RamseyState {
    const ACTION: usize = ACTION;
    fn action_rewards(&self) -> Vec<(usize, f32)> {
        let Self {
            colors: _,
            edges: _,
            neighborhoods: _,
            available_actions: _,
            ordered_actions: OrderedEdgeRecolorings(ordered_actions),
            counts: _,
            time_remaining: _,
        } = self;
        ordered_actions
            .iter()
            .map(|(recolor, reward)| {
                let EdgeRecoloring {
                    new_color,
                    edge_position,
                } = recolor;
                let action_index = new_color * E + edge_position;
                (action_index, *reward as f32)
            })
            .collect()
    }

    // todo! lol
    fn act(&mut self, action: usize) {
        let Self {
            colors: ColoredCompleteGraph(colors),
            edges: MulticoloredGraphEdges(edges),
            neighborhoods: MulticoloredGraphNeighborhoods(neighborhoods),
            available_actions,
            ordered_actions: OrderedEdgeRecolorings(ordered_actions),
            counts: CliqueCounts(counts),
            time_remaining,
        } = self;
        let new_uv_color = action / E;
        let edge_position = action % E;
        // dbg!(new_uv_color, edge_position);
        let Color(old_uv_color) = colors[edge_position];
        // dbg!(old_uv_color);
        let (u, v) = edge_from_position(edge_position);
        // dbg!(u, v);
        // assert_ne!(u, v, "{edge_position}");

        // (0..C).for_each(|c| {
        //     (0..N).for_each(|u| {
        //         let neigh = neighborhoods[c][u];
        //         assert_eq!(neigh & (1 << u), 0, "u = {u}, neigh = {neigh:b}");
        //     });
        // });

        /* when does k(xy, c) change?
            necessarily, c is either old_color or new_color
            k(uv, c) never changes
            k(uw, c) and k(vw, c) may change
            k(wx, c) may also change
        */

        /* when does k = k(uw, c_old) change? (w != u, v)
            k counts the number of quadruples Q = {u, w, x, y} s.t.
                G_{c_old}[Q] + {uw} is a clique
            with this recoloring, Q is no longer a clique iff
                Q = {u, w, v, x} for some x
                w is in N_{c_old}(v) \ {u}
                x is in N_{c_old}(u) & N_{c_old}(w) & (N_{c_old}(v) \ {u})
                since x is not in N_c(u), we may *not* omit the `\ {u}`
        */
        // after updating the counts, we update all affected values of r(xy, c)
        // r(uv, c) is unaffected (in fact, these values are removed)
        // we store which edges wx have an affected column and update them later
        // todo!() this can be optimized by only updating the affected columns within the iterator
        let mut affected_count_columns: Vec<usize> = vec![];

        let old_neigh_u = neighborhoods[old_uv_color][u];
        let old_neigh_v = neighborhoods[old_uv_color][v];

        // we remove v and u so that they are not treated as w or x in the following
        let old_neigh_u = old_neigh_u ^ (1 << v);
        let old_neigh_v = old_neigh_v ^ (1 << u);
        let old_neigh_uv = old_neigh_u & old_neigh_v;

        // assert_eq!(old_neigh_u & (1 << u), 0, "u = {u}, v = {v}, old_neigh_u = {old_neigh_u:b}");
        // assert_eq!(old_neigh_v & (1 << v), 0, "u = {u}, v = {v}, old_neigh_v = {old_neigh_v:b}");

        BitIter::from(old_neigh_v).for_each(|w| {
            let old_neigh_w = neighborhoods[old_uv_color][w];
            let old_neigh_uvw = old_neigh_uv & old_neigh_w;
            let k_uw_old_decrease = old_neigh_uvw.count_ones();
            if k_uw_old_decrease != 0 {
                // decrease k(uw, c_uv_old)
                let uw_position = edge_to_position(u, w);
                counts[old_uv_color][uw_position] -= k_uw_old_decrease as i32;
                /* when does r = r(e, c) change?
                    consider r(e, c) = k(e, c_e) - k(e, c)
                        here, c_e is the current color of e
                        r(e, c) is only defined when c_e != c
                    w.l.o.g., e = uw
                    Case I: c_uw = c_uv_old
                        r(uw, c) = k(uw, c_uv_old) - k(uw, c)
                        assumes that c != c_uv_old
                        so r(uw, c) deecreases by the same decrease to k(uw, c_uv_old)
                    Case II: c_uw
                        todo!("adjust all affected values of r(uw, c)")
                */
                affected_count_columns.push(uw_position);
            }
        });
        BitIter::from(old_neigh_u).for_each(|w| {
            let old_neigh_w = neighborhoods[old_uv_color][w];
            let old_neigh_uvw = old_neigh_uv & old_neigh_w;
            let k_vw_old_decrease = old_neigh_uvw.count_ones();
            if k_vw_old_decrease != 0 {
                // decrease k(vw, c_old)
                let vw_position = edge_to_position(v, w);
                // assert!(vw_position <= 135, "(v, w) = ({v}, {w})");
                counts[old_uv_color][vw_position] -= k_vw_old_decrease as i32;
                // todo!("adjust all affected values of r(vw, c)")
                affected_count_columns.push(vw_position);
            }
        });
        /* when does k = k(wx, c_old) change? (w, x != u, v)
            k counts the number of quadruples Q = {w, x, u', v'} s.t.
                G_{c_old}[Q] + {wx} is a clique
            with this recoloring, Q is no longer a clique iff
                Q = {w, x, u, v}
                w, x are in N_{c_old}(u) & N_{c_old}(v)
        */
        BitIter::from(old_neigh_uv)
            .tuple_combinations()
            .for_each(|(w, x)| {
                let wx_position = edge_to_position(w, x);
                counts[old_uv_color][wx_position] -= 1;
                // todo!("adjust all affected values of r(wx, c)")
                affected_count_columns.push(wx_position);
            });

        // we do not need to remove v and u -- uv has color old_uv_color
        let new_neigh_u = neighborhoods[new_uv_color][u];
        // dbg!(format!("{new_neigh_u:b}"));
        let new_neigh_v = neighborhoods[new_uv_color][v];
        // dbg!(format!("{new_neigh_v:b}"));
        let new_neigh_uv = new_neigh_u & new_neigh_v;
        // dbg!(format!("{new_neigh_uv:b}"));

        // assert_ne!(new_neigh_u & (1 << u), 0, "u = {u}, v = {v}, new_neigh_u = {new_neigh_u:b}");
        // assert_ne!(new_neigh_v & (1 << v), 0, "u = {u}, v = {v}, new_neigh_v = {new_neigh_v:b}");

        BitIter::from(new_neigh_v).for_each(|w| {
            let new_neigh_w = neighborhoods[new_uv_color][w];
            let new_neigh_uvw = new_neigh_uv & new_neigh_w;
            let k_uw_new_increase = new_neigh_uvw.count_ones();
            if k_uw_new_increase != 0 {
                // decrease k(uw, c_uv_old)
                let uw_position = edge_to_position(u, w);
                // assert!(uw_position <= 135, "(u, w) = ({u}, {w})");
                counts[new_uv_color][uw_position] += k_uw_new_increase as i32;
                /* when does r = r(e, c) change?
                    consider r(e, c) = k(e, c_e) - k(e, c)
                        here, c_e is the current color of e
                        r(e, c) is only defined when c_e != c
                    w.l.o.g., e = uw
                    Case I: c_uw = c_uv_old
                        r(uw, c) = k(uw, c_uv_old) - k(uw, c)
                        assumes that c != c_uv_old
                        so r(uw, c) deecreases by the same decrease to k(uw, c_uv_old)
                    Case II: c_uw
                        todo!("adjust all affected values of r(uw, c)")
                */
                affected_count_columns.push(uw_position);
            }
        });
        BitIter::from(new_neigh_u).for_each(|w| {
            let new_neigh_w = neighborhoods[new_uv_color][w];
            let new_neigh_uvw = new_neigh_uv & new_neigh_w;
            let k_vw_new_increase = new_neigh_uvw.count_ones();
            if k_vw_new_increase != 0 {
                // decrease k(vw, c_old)
                let vw_position = edge_to_position(v, w);
                // assert!(uw_position <= 135, "(u, w) = ({u}, {w})");
                counts[new_uv_color][vw_position] += k_vw_new_increase as i32;
                // todo!("adjust all affected values of r(vw, c)")
                affected_count_columns.push(vw_position);
            }
        });
        /* when does k = k(wx, c_old) change? (w, x != u, v)
            k counts the number of quadruples Q = {w, x, u', v'} s.t.
                G_{c_old}[Q] + {wx} is a clique
            with this recoloring, Q is no longer a clique iff
                Q = {w, x, u, v}
                w, x are in N_{c_old}(u) & N_{c_old}(v)
        */
        BitIter::from(new_neigh_uv)
            .tuple_combinations()
            .for_each(|(w, x)| {
                let wx_position = edge_to_position(w, x);
                counts[new_uv_color][wx_position] += 1;
                // todo!("adjust all affected values of r(wx, c)")
                affected_count_columns.push(wx_position);
            });

        affected_count_columns.into_iter().for_each(|wx_position| {
            let Color(wx_color) = colors[wx_position];
            let old_count = counts[wx_color][wx_position];
            // update r(wx, c) for all c != wx_color
            (0..C).for_each(|c| {
                let reward = old_count - counts[c][wx_position];
                let recoloring = EdgeRecoloring {
                    new_color: c,
                    edge_position: wx_position,
                };
                let _old_reward = ordered_actions.change_priority(&recoloring, reward);
                // todo!("update all affected count columns");
            });
        });

        colors[edge_position] = Color(new_uv_color);
        edges[old_uv_color][edge_position] = false;
        edges[new_uv_color][edge_position] = true;
        neighborhoods[old_uv_color][u] ^= 1 << v;
        neighborhoods[old_uv_color][v] ^= 1 << u;
        neighborhoods[new_uv_color][u] ^= 1 << v;
        neighborhoods[new_uv_color][v] ^= 1 << u;

        (0..C).for_each(|c| {
            available_actions[c][edge_position] = false;
        });
        (0..C).for_each(|c| {
            let recoloring = EdgeRecoloring {
                new_color: c,
                edge_position,
            };
            ordered_actions.remove(&recoloring);
        });
        *time_remaining -= 1;
    }

    fn is_terminal(&self) -> bool {
        let Self {
            colors: _,
            edges: _,
            neighborhoods: _,
            available_actions: _,
            ordered_actions: _,
            counts: _,
            time_remaining,
        } = self;
        *time_remaining == 0
    }

    fn reset(&mut self, time: usize) {
        let Self {
            colors: ColoredCompleteGraph(colors),
            edges: _,
            neighborhoods: _,
            available_actions,
            ordered_actions: OrderedEdgeRecolorings(ordered_actions),
            counts: CliqueCounts(counts),
            time_remaining,
        } = self;
        /* when resetting:
            action (uv, c) is available <==> uv is not colored c
            ordered_actions orders by r = k(uv, c_uv) - k(uv, c)
                where c_uv is the current color of uv
            time_remaining is the given time
        */
        // update available actions and ordered actions
        *available_actions = [[true; E]; C];
        colors.iter().enumerate().for_each(|(i, c)| {
            let Color(old_color) = c;
            available_actions[*old_color][i] = false;
            let old_count = counts[*old_color][i];
            (0..C).filter(|c| old_color.ne(c)).for_each(|new_color| {
                let new_count = counts[new_color][i];
                let reward = old_count - new_count;
                let recoloring = EdgeRecoloring {
                    new_color,
                    edge_position: i,
                };
                let _old_reward = ordered_actions.push(recoloring, reward);
            });
        });
        // update time_remaining
        *time_remaining = time;
    }

    fn to_vec(&self) -> [f32; STATE] {
        self.to_vec()
    }
}

impl Cost for RamseyState {
    fn cost(&self) -> f32 {
        let Self {
            colors: ColoredCompleteGraph(colors),
            edges: _,
            neighborhoods: _,
            available_actions: _,
            ordered_actions: _,
            counts: CliqueCounts(counts),
            time_remaining: _,
        } = self;
        let count = colors
            .iter()
            .enumerate()
            .map(|(i, c)| {
                let Color(c) = c;
                counts[*c][i]
            })
            .sum::<i32>()
            / 6;
        count as f32
    }
}

#[cfg(test)]
mod test {
    use super::*;

    fn incident_cliques(s: &RamseyState, pos: usize) -> [Vec<(usize, usize)>; C] {
        let RamseyState {
            colors: _,
            edges: _,
            neighborhoods: MulticoloredGraphNeighborhoods(neighborhoods),
            available_actions: _,
            ordered_actions: _,
            counts: _,
            time_remaining: _,
        } = s;
        let mut incident_cliques: [Vec<(usize, usize)>; C] = core::array::from_fn(|_| vec![]);
        let (u, v) = edge_from_position(pos);
        let other_vertices = (0..N).filter(|w| u.ne(w) && v.ne(w));
        other_vertices.tuple_combinations().for_each(|(w, x)| {
            (0..C).for_each(|c| {
                // check for the edges uw, ux, vw, vx, and wx
                let neigh_uv = neighborhoods[c][u] & neighborhoods[c][v];
                if neigh_uv & (1 << w) != 0 && neigh_uv & (1 << w) != 0 {
                    if neighborhoods[c][w] & (1 << x) != 0 {
                        incident_cliques[c].push((w, x));
                    }
                }
            })
        });
        incident_cliques
    }

    #[test]
    fn the_all_red_graph_has_the_correct_graph_state() {
        let mut state = RamseyState::all_red();
        // let GraphState {
        //     colors: ColoredCompleteGraph(colors),
        //     edges: MulticoloredGraphEdges(edges),
        //     neighborhoods: MulticoloredGraphNeighborhoods(neighborhoods),
        //     available_actions: _,
        //     ordered_actions: _,
        //     counts: _,
        //     time_remaining,
        // } = s;
        let cliques = incident_cliques(&state, 0);
        assert_eq!(cliques[0].len(), 105);
        (1..C).for_each(|i| assert_eq!(cliques[i].len(), 0));
        assert_eq!(state.cost(), 2380.0);
        let rewards = state.action_rewards();
        assert_eq!(rewards.len(), 136);
        let (_, reward) = rewards.into_iter().find(|(a, _)| *a == E).unwrap();
        assert_eq!(reward, 105.0);
        state.act(E);
        assert_eq!(state.cost(), 2275.0);
    }
}
