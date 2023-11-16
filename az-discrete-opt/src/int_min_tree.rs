// use rayon::prelude::{IntoParallelIterator, ParallelIterator};
use std::{collections::BTreeMap, mem::MaybeUninit};

// use core::num::NonZeroUsize;
use crate::{state::cost::Cost, space::StateActionSpace, tree_node::{TreeNode, TreeNodeFor}, path::ActionPathFor};

pub struct INTMinTree<P> {
    root_data: INTStateData,
    data: Vec<BTreeMap<P, StateDataKind>>,
}

pub enum StateDataKind {
    Exhausted { c_t_star: f32 },
    Active { data: INTStateData },
}

enum EndNodeAndLevel<'a, P> {
    NewNodeNewLevel,
    NewNodeOldLevel(&'a mut BTreeMap<P, StateDataKind>),
    OldExhaustedNode { c_t_star: f32 },
}

struct INTTransition<'a> {
    data_i: &'a mut INTStateData,
    a_i_plus_one: usize,
}

impl<P> INTMinTree<P> {
    pub fn new<Space>(
        root_predictions: &[f32],
        cost: f32,
        root: &Space::State,
    ) -> Self
    where
        Space: StateActionSpace,
    {
        Self {
            root_data: INTStateData::new::<Space>(root_predictions, cost, root),
            data: Vec::new(),
        }
    }

    // pub fn par_new_trees<Space, const B: usize, const A: usize, C>(
    //     prob_0: &[[f32; A]; B],
    //     c_0: &[C; B],
    //     s_0: &[Space::State; B],
    // ) -> [Self; B]
    // where
    //     Space: StateActionSpace,
    //     Space::State: Sync,
    //     Self: Send,
    //     C: Cost<f32> + Sync,
    //     // N: TreeNode<State = Space::State, Path = P> + Sync,
    //     // N::Action: Action<N>,
    // {
    //     let mut trees: [MaybeUninit<Self>; B] = MaybeUninit::uninit_array();
    //     (&mut trees, prob_0, c_0, s_0)
    //         .into_par_iter()
    //         .for_each(|(t, root_predictions, cost, root)| {
    //             t.write(Self::new::<Space>(root_predictions, cost.cost(), root));
    //         });
    //     unsafe { MaybeUninit::array_assume_init(trees) }
    // }

    // pub fn replant<N>(&mut self, root_predictions: &[f32], cost: f32, root: &S)
    // where
    //     N: TreeNode<Path = P>,
    //     N::Action: Action<N>,
    // {
    //     self.data.clear();
    //     self.root_data = INTStateData::new(root_predictions, cost, root);
    // }

    pub fn simulate_once<'a, Space>(
        &'a mut self,
        n_0: &mut (impl TreeNode<Path = P> + TreeNodeFor<Space>),
    ) -> INTTransitions<'a, P>
    where
        // Space::Action: Action<Space>,
        Space: StateActionSpace,
        P: ActionPathFor<Space> + Ord,
        // N: TreeNode<Path = P> + TreeNodeFor<Space>,
        // N: TreeNode<Path = P>,
        // N::Action: Action<N>,
        // N::Path: Ord,
    {
        let Self { root_data, data } = self;
        let a_1 = root_data.best_action();
        let action_1 = Space::from_index(a_1);
        let n_i = n_0;
        n_i.apply_action(&action_1);
        // n_i.apply_action(&action_1);
        // unsafe { n_i.state().act_unchecked(&action_1) };
        // let p_i = p_0;
        // p_i.push(&action_1);
        let mut transitions: Vec<INTTransition> = vec![];

        for (_depth, data) in data.iter_mut().enumerate() {
            // Polonius case III: https://github.com/rust-lang/rfcs/blob/master/text/2094-nll.md#problem-case-3-conditional-control-flow-across-functions
            /* isomorphic to
            enum PreviouslyExhaustedValue {
                NotFound,
                FoundActive,
                // the borrow checker doesn't do inference down branches of enums, it processes ownership as a stack
                FoundExhausted { c_t_star: f32 },
            }
            */
            let previously_exhausted_value: Option<Option<f32>> =
                data.get(n_i.path()).map(|data| match data {
                    StateDataKind::Exhausted { c_t_star } => Some(*c_t_star),
                    StateDataKind::Active { data: _ } => None,
                });
            match previously_exhausted_value {
                Some(Some(c_t_star)) => {
                    let end = EndNodeAndLevel::OldExhaustedNode { c_t_star };
                    return INTTransitions {
                        a_1,
                        root_data,
                        transitions,
                        end,
                    };
                }
                Some(None) => {}
                None => {
                    let end = EndNodeAndLevel::NewNodeOldLevel(data);
                    return INTTransitions {
                        a_1,
                        root_data,
                        transitions,
                        end,
                    };
                }
            }
            let state_data = match data.get_mut(n_i.path()) {
                Some(StateDataKind::Active { data }) => data,
                _ => unreachable!("this should be unreachable"),
            };
            let a_i_plus_one = state_data.best_action();
            let action_i_plus_1 = Space::from_index(a_i_plus_one);

            // dbg!(a_i_plus_one);
            debug_assert_eq!(Space::index(&action_i_plus_1), a_i_plus_one);
            // debug_assert!(
            //     n_i.state().actions().any(|a| action_i_plus_1 == a),
            //     "self = {}, action_i_plus_1 = {action_i_plus_1}, actions = {:?}\ndepth = {_depth}",
            //     n_i.state(),
            //     n_i.state().actions().map(|a| a.index()).collect::<Vec<_>>(),
            // );

            // debug_assert!(
            //     s_i.actions().any(|a| action_i_plus_1 == a),
            //     "self = {s_i}, action_i_plus_1 = {action_i_plus_1}, actions = {:?}\np_i = {p_i:?}\ndepth = {_depth}",
            //     s_i.actions().map(|a| a.index()).collect::<Vec<_>>(),
            // );

            transitions.push(INTTransition {
                data_i: state_data,
                a_i_plus_one,
            });
            n_i.apply_action(&action_i_plus_1);
        }
        INTTransitions {
            a_1,
            transitions,
            // p_t: p_i,
            root_data,
            end: EndNodeAndLevel::NewNodeNewLevel,
        }
    }

    // todo! refactor/deprecate
    // pub(crate) fn _par_simulate_once<'a, Space, N, const B: usize>(
    //     trees: &'a mut [Self; B],
    //     n_0: &mut [N; B],
    // ) -> [INTTransitions<'a, P>; B]
    // where
    //     Space: StateActionSpace,
    //     P: ActionPathFor<Space> + Ord,
    //     Self: Send,
    //     N: TreeNode<Path = P> + TreeNodeFor<Space> + Send,
    //     P: Ord,
    //     INTTransitions<'a, P>: Send,
    // {
    //     let mut trans: [MaybeUninit<INTTransitions<P>>; B] = MaybeUninit::uninit_array();
    //     // let foo = trans.into_par_iter();
    //     (&mut trans, trees, n_0)
    //         .into_par_iter()
    //         .for_each(|(trans, t, n_0)| {
    //             trans.write(t.simulate_once(n_0));
    //         });
    //     unsafe { MaybeUninit::array_assume_init(trans) }
    // }

    pub fn insert_node_at_next_level(
        &mut self,
        un_t: UpdatedNode,
        p_t: &P,
    )
    where
        P: Ord + Clone,
    {
        let level = BTreeMap::from([(p_t.clone(), un_t.data_t)]);
        self.data.push(level);
    }

    pub fn observe(&self, probs: &mut [f32], values: &mut [f32]) {
        probs.fill(0.0);
        debug_assert_eq!(values.len(), 1);
        let Self { root_data, data: _ } = self;
        root_data.observe(probs, values);
    }
}

pub trait INTState {
    fn act(&mut self, action: usize);
    fn is_terminal(&self) -> bool;
    // fn update_vec(&self, state_vec: &mut [f32]);
    fn actions(&self) -> Vec<usize>;
}

pub struct INTTransitions<'a, P> {
    a_1: usize,
    root_data: &'a mut INTStateData,
    transitions: Vec<INTTransition<'a>>,
    // p_t: P,
    end: EndNodeAndLevel<'a, P>,
}

pub struct UpdatedNode {
    // p_t: P,
    data_t: StateDataKind,
}

impl<'a, P> INTTransitions<'a, P> {
    pub fn update_existing_nodes<N, Space>(
        self,
        c_t: &impl Cost<f32>,
        s_t: &Space::State,
        // s_t: &impl State,
        p_t: &P,
        probs_t: &[f32],
        g_star_theta_s_t: &[f32],
    ) -> Option<UpdatedNode>
    where
        Space: StateActionSpace,
        P: ActionPathFor<Space> + Ord,
        N: TreeNode<Path = P, State = Space::State>,
        P: Ord + Clone,
        // N::Action: Action<N>,
    {
        debug_assert_eq!(g_star_theta_s_t.len(), 1);
        let INTTransitions {
            a_1,
            transitions,
            // p_t,
            root_data,
            end,
        } = self;
        let (h_star_theta_s_t, node): (f32, _) = match end {
            EndNodeAndLevel::NewNodeNewLevel => {
                // let data_t = INTStateData::new(probs_t, c_t.cost(), s_t);
                let (h_star_theta_s_t, data_t) = match Space::is_terminal(s_t) {
                    true => (
                        0.0,
                        StateDataKind::Exhausted {
                            c_t_star: c_t.cost(),
                        },
                    ),
                    false => (
                        g_star_theta_s_t[0],
                        StateDataKind::Active {
                            data: INTStateData::new::<Space>(probs_t, c_t.cost(), s_t),
                        },
                    ),
                };
                let node = UpdatedNode { /* p_t,*/ data_t };
                (h_star_theta_s_t, Some(node))
            }
            EndNodeAndLevel::NewNodeOldLevel(t) => {
                let data = match Space::is_terminal(s_t) {
                    true => StateDataKind::Exhausted {
                        c_t_star: c_t.cost(),
                    },
                    false => {
                        let data = INTStateData::new::<Space>(probs_t, c_t.cost(), s_t);
                        StateDataKind::Active { data }
                    }
                };
                // t.insert(p_t, data).unwrap();
                let _i = t.insert(p_t.clone(), data);
                debug_assert!(_i.is_none(), "this node should not already exist");
                // todo!();
                // let node = UnupdatedNode::OldLevel(t.get_mut(&p_t).unwrap());
                let h_star_theta_s_t = match Space::is_terminal(s_t) {
                    true => 0.0,
                    false => g_star_theta_s_t[0],
                };
                (h_star_theta_s_t, None)
            }
            EndNodeAndLevel::OldExhaustedNode { c_t_star: _ } => {
                // todo! we can be clever and use the fact that we know the cost of all nodes reachable from the exhausted node
                (0.0, None)
            }
        };

        // todo! also backpropagate exhaustion
        let mut c_star_theta_i = c_t.cost() - h_star_theta_s_t.max(0.0);
        transitions.into_iter().rev().for_each(|t_i| {
            let INTTransition {
                data_i,
                // c_i,
                a_i_plus_one,
            } = t_i;
            // let g_star_theta_i = c_i - c_star_theta_i;
            let c_i = data_i.c_s;
            data_i.update(a_i_plus_one, &mut c_star_theta_i);
            c_star_theta_i = c_star_theta_i.min(c_i);
            // ");
        });
        root_data.update(a_1, &mut c_star_theta_i);
        node
    }
}

/* todo! refactor so that:
    `actions` = [a0, ..., a_{k-1}, a_k, ..., a_{n-1}]
        here, a0, ..., a_{k-1} are visited, the rest unvisited
        when initialized, we sort by probability
        when an action is visited for the first time, we increment the counter k
        visited actions are sorted by upper estimate
        when selecting the next action, we compare the best visited action to the best unvisited action
        unvisited actions use the same upper estimate formula, but it depends only on the probability
*/
#[derive(Debug)]
pub struct INTStateData {
    n_s: usize,
    c_s: f32,
    actions: Vec<INTActionData>,
}

impl INTStateData {
    fn observe(&self, probs: &mut [f32], values: &mut [f32]) {
        probs.fill(0.0);
        debug_assert_eq!(values.len(), 1);
        let Self {
            n_s,
            c_s: _,
            actions,
        } = self;
        let n_s = *n_s as f32;
        let mut value = 0.0;
        actions.iter().filter(|a| a.n_sa != 0).for_each(|a| {
            let n_sa = a.n_sa as f32;
            let q_sa = a.g_sa_sum / n_sa;
            probs[a.a] = n_sa / n_s;
            value += q_sa;
        });
        values[0] = value / n_s;
    }

    fn __actions(&self) -> Vec<usize> {
        let mut actions = self.actions.iter().map(|a| a.a).collect::<Vec<_>>();
        actions.sort();
        actions
    }

    // todo! refactor to `Option<Self>` and cut the `is_terminal` checks from `simulate_once`
    pub fn new<Space>(
        probs: &[f32],
        cost: f32,
        state: &Space::State,
    ) -> Self
    where
        Space: StateActionSpace,
        // N: TreeNode<State = Space::State>,
        // N::Action: Action<N>,
    {
        let p_sum = Space::actions(state).map(|a| probs[a]).sum::<f32>();// = N::Action::actions(state).map(|a| probs[a]).sum::<f32>();
        let mut actions = Space::actions(state)
            .map(|a| {
                let p = probs[a] / p_sum;
                INTActionData::new(a, p)
            })
            .collect::<Vec<_>>();
        actions.sort_by(|a, b| b.u_sa.total_cmp(&a.u_sa));
        Self {
            n_s: 0,
            c_s: cost,
            actions,
        }
    }

    pub fn best_action(&self) -> usize {
        let Self {
            n_s: _,
            c_s: _,
            actions,
        } = self;
        actions[0].a
    }

    pub fn update(&mut self, a_i_plus_one: usize, c_star_theta_i_plus_one: &mut f32) {
        let Self { n_s, c_s, actions } = self;
        let g_star_theta_i = *c_s - *c_star_theta_i_plus_one;
        *n_s += 1;
        let action_data = actions.iter_mut().find(|a| a.a == a_i_plus_one).unwrap();
        action_data.update(g_star_theta_i);
        self.update_upper_estimates();
        self.sort_actions();
    }

    fn update_upper_estimates(&mut self) {
        let Self {
            n_s,
            c_s: _,
            actions,
        } = self;
        actions
            .iter_mut()
            .for_each(|a| a.update_upper_estimate(*n_s));
    }

    fn sort_actions(&mut self) {
        let Self {
            n_s: _,
            c_s: _,
            actions,
        } = self;
        actions.sort_by(|a, b| b.u_sa.total_cmp(&a.u_sa));
    }
}

#[derive(Debug)]
struct INTActionData {
    a: usize,
    p_sa: f32,
    n_sa: usize,
    g_sa_sum: f32,
    u_sa: f32,
}

const C_PUCT_0: f32 = 0.5;
const C_PUCT: f32 = 0.5;

impl INTActionData {
    pub(crate) fn new(a: usize, p_a: f32) -> Self {
        Self {
            a,
            p_sa: p_a,
            n_sa: 0,
            g_sa_sum: 0.0,
            u_sa: C_PUCT_0 * p_a,
        }
    }

    pub(crate) fn update(&mut self, g_star_theta_i: f32) {
        let Self {
            a: _,
            p_sa: _,
            n_sa,
            g_sa_sum,
            u_sa: _,
        } = self;
        *n_sa += 1;
        *g_sa_sum += g_star_theta_i;
    }

    pub(crate) fn update_upper_estimate(&mut self, n_s: usize) {
        let Self {
            a: _,
            p_sa,
            n_sa,
            g_sa_sum,
            u_sa,
        } = self;
        let n_sa = (*n_sa).max(1) as f32;
        let q_sa = *g_sa_sum / n_sa;
        let n_s = n_s as f32;
        let p_sa = *p_sa;
        *u_sa = q_sa / n_sa + C_PUCT * p_sa * (n_s.sqrt() / n_sa);
    }
}
