use std::{collections::BTreeMap, mem::MaybeUninit};
use rayon::prelude::{IntoParallelIterator, ParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator};

// use core::num::NonZeroUsize;
use crate::{iq_min_tree::ActionsTaken, state::{State, Action, cost::Cost}};

pub struct INTMinTree {
    root_data: INTStateData,
    data: BTreeMap<ActionsTaken, INTStateData>,
}

pub trait __INTStateDiagnostic {
    fn __actions(&self) -> Vec<usize>;
}

struct INTTransition {
    p_i: ActionsTaken,
    c_i: f32,
    a_i_plus_one: usize,
}

impl INTMinTree {
    pub fn new<S: State>(root_predictions: &[f32], cost: f32, root: &S) -> Self {
        Self {
            root_data: INTStateData::new(root_predictions, cost, root),
            data: BTreeMap::new(),
        }
    }

    pub fn par_new_trees<S: State, const N: usize, const A: usize, C>(
        root_predictions: &[[f32; A]; N],
        costs: &[C; N],
        roots: &[S; N],
    ) -> [Self; N]
    where
        S: Send + Sync + Clone,
        C: Sync + Cost<f32>,
        Self: Send,
    {
        let mut trees: [MaybeUninit<Self>; N] = MaybeUninit::uninit_array();
        (&mut trees, root_predictions, costs, roots).into_par_iter().for_each(|(t, root_predictions, cost, root)| {
            t.write(Self::new(root_predictions, cost.cost(), root));
        });
        unsafe { MaybeUninit::array_assume_init(trees) }
    }

    pub fn replant<S: State>(&mut self, root_predictions: &[f32], cost: f32, root: &S) {
        self.data.clear();
        self.root_data = INTStateData::new(root_predictions, cost, root);
    }

    pub fn simulate_once<S>(&self, s_0: &mut S) -> INTTransitions
    where
        S: State + core::fmt::Display, //+ Cost + core::fmt::Display + __INTStateDiagnostic
        S::Actions: Eq + core::fmt::Display,
    {
        let Self { root_data, data } = self;
        // let mut __state_actions = s_0.__actions();
        // __state_actions.sort();
        // dbg!();
        // assert_eq!(__state_actions, root_data.__actions());
        // assert_eq!(s_0.cost(), root_data.c_s);
        // dbg!();
        // println!("root = \n{s_0}");
        // s_0.actions().for_each(|a| {
        //     println!("a = {a}");
        // });
        let a_1 = root_data.best_action();
        let action_1 = unsafe { S::Actions::from_index_unchecked(a_1) };
        // println!("a1, action_1 = {a_1}, {action_1}");
        // todo!();
        let s_i = s_0;
        unsafe { s_i.act_unchecked(&action_1) };
        let mut p_i = ActionsTaken::new(a_1);
        let mut transitions: Vec<INTTransition> = vec![];
        while !s_i.is_terminal() {
            // dbg!();
            // println!("s_i = {s_i}");
            // s_i.actions().for_each(|a| {
            //     println!("a = {a}");
            // });
            if let Some(data) = data.get(&p_i) {
                // assert_eq!(s_0.cost(), data.c_s);
                // let mut __state_actions = s_0.__actions();
                // __state_actions.sort();
                // assert_eq!(__state_actions, data.__actions());
                let a_i_plus_one = data.best_action();
                // dbg!(a_i_plus_one);
                transitions.push(INTTransition {
                    p_i: p_i.clone(),
                    c_i: data.c_s,
                    a_i_plus_one,
                });
                let action_i_plus_1 = unsafe { S::Actions::from_index_unchecked(a_i_plus_one) };
                // println!("a_i_plus_one, action_i = {a_i_plus_one}, {action_i_plus_1}");
                // dbg!();
                s_i.act(&action_i_plus_1);
                // dbg!();
                // dbg!();
                p_i.push(a_i_plus_one);
                // dbg!();
            } else {
                // dbg!();
                return INTTransitions {
                    a_1,
                    transitions,
                    p_t: INTSearchEnd::Unvisited { state_path: p_i },
                }
            }
        }
        // dbg!();
        INTTransitions {
            a_1,
            transitions,
            p_t: INTSearchEnd::Terminal { state_path: p_i },
        }
        // INTTransitions {
        //     first_action,
        //     transitions,
        //     end: INTSearchEnd::Unvisited { state_path },
        // }
    }

    pub fn par_simulate_once<S, const N: usize>(
        trees: &[Self; N],
        s_0: &mut [S; N],
    ) -> [INTTransitions; N]
    where
        S: Send + core::fmt::Display + core::fmt::Debug + State,
        S::Actions: Eq + core::fmt::Display,
        // S: State + core::fmt::Display, //+ Cost + core::fmt::Display + __INTStateDiagnostic
        // S::Actions: Eq + core::fmt::Display,
    {
        let mut trans: [MaybeUninit<INTTransitions>; N] = MaybeUninit::uninit_array();
        // let foo = trees.par_iter();
        // let bar = s_0.par_iter_mut();
        (&mut trans, trees, s_0).into_par_iter().for_each(|(trans, t, s)| {
            trans.write(t.simulate_once(s));
        });
        unsafe { MaybeUninit::array_assume_init(trans) }
    }

    pub fn insert<S, C>(
        &mut self,
        transitions: &INTTransitions,
        s_t: &S,
        c_t: &C,
        prob_s_t: &[f32],
    )
    where
        S: State + core::fmt::Display,
        S::Actions: core::fmt::Display,
        C: Cost<f32>,
    {
        // dbg!();
        // println!("inserting s_t = {s_t}\nwhich has actions:");
        // s_t.actions().for_each(|a| {
        //     println!("a = {a}");
        // });
        let Self { root_data: _, data } = self;
        let p_t = transitions.last_path();
        let state_data = INTStateData::new(prob_s_t, c_t.cost(), s_t);
        // dbg!(&state_data);
        data.insert(p_t.clone(), state_data);
        // panic!();
    }

    pub fn update<C>(
        &mut self,
        transitions: &INTTransitions,
        c_t: &C,
        g_star_theta_s_t: &[f32],
    )
    where
        C: Cost<f32>,
    {
        assert_eq!(g_star_theta_s_t.len(), 1);
        let INTTransitions {
            a_1,
            transitions,
            p_t,
        } = transitions;
        let h_star_theta_s_t = match p_t {
            INTSearchEnd::Terminal { .. } => 0.0,
            INTSearchEnd::Unvisited { .. } => g_star_theta_s_t[0],
        };
        // we run from i = t to i = 0
        let mut c_star_theta_i = c_t.cost() - h_star_theta_s_t.max(0.0);
        let Self { root_data, data } = self;
        transitions.into_iter().rev().for_each(|t_i| {
            let INTTransition { p_i, c_i, a_i_plus_one } = t_i;
            // let g_star_theta_i = c_i - c_star_theta_i;
            let data_i = data.get_mut(p_i).unwrap();
            data_i.update(*a_i_plus_one, &mut c_star_theta_i);
            c_star_theta_i = c_star_theta_i.min(*c_i);
        });
        root_data.update(*a_1, &mut c_star_theta_i);
    }

    pub fn observe(&self, probs: &mut [f32], values: &mut [f32]) {
        probs.fill(0.0);
        assert_eq!(values.len(), 1);
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

pub struct INTTransitions {
    a_1: usize,
    transitions: Vec<INTTransition>,
    p_t: INTSearchEnd,
}

impl INTTransitions {
//     pub fn last_cost(&self) -> Option<f32> {
//         todo!()
//     }

    pub fn last_path(&self) -> &ActionsTaken {
        match &self.p_t {
            INTSearchEnd::Terminal { state_path, .. } => state_path,
            INTSearchEnd::Unvisited { state_path, .. } => state_path,
        }
    }
}

pub(crate) enum INTSearchEnd {
    Terminal { state_path: ActionsTaken, },
    Unvisited { state_path: ActionsTaken },
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
struct INTStateData {
    n_s: usize,
    c_s: f32,
    actions: Vec<INTActionData>,
}

impl INTStateData {
    fn observe(&self, probs: &mut [f32], values: &mut [f32]) {
        probs.fill(0.0);
        assert_eq!(values.len(), 1);
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
    
    pub fn new<S: State>(predctions: &[f32], cost: f32, state: &S) -> Self {
        let p_sum = state.actions().map(|a| predctions[a.index()]).sum::<f32>();
        let mut actions = state.actions().into_iter().map(|a| {
            let p = predctions[a.index()] / p_sum;
            INTActionData::new(a.index(), p)
        }).collect::<Vec<_>>();
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
        // if self.n_s > 4 {
        //     dbg!(actions);
        //     panic!();
        // }
        actions[0].a
    }

    pub fn update(&mut self, a_i_plus_one: usize, c_star_theta_i_plus_one: &mut f32) {
        let Self {
            n_s,
            c_s,
            actions,
        } = self;
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
        actions.iter_mut().for_each(|a| a.update_upper_estimate(*n_s));
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

const C_PUCT_0: f32 = 10.0;
const C_PUCT: f32 = 10.0;

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
