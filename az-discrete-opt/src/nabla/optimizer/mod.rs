pub use petgraph::stable_graph::NodeIndex;

use crate::log::ArgminData;

use super::{model::NablaModel, space::DfaWithCost, tree::SearchTree};

pub struct NablaOptimizer<Space: DfaWithCost, M, P> {
    space: Space,
    roots: Vec<Space::State>,
    states: Vec<Space::State>,
    action_bools: Vec<bool>,
    num_actions: Vec<f32>,
    costs: Vec<Space::Cost>,
    paths: Vec<P>,
    // transitions: Vec<Vec<Transition>>,
    last_positions: Vec<NodeIndex>,
    state_vecs: Vec<f32>,
    v_host: Vec<f32>,
    p_host: Vec<f32>,
    // action_weights: Vec<f32>,
    trees: Vec<SearchTree>,
    model: M,
    num_inspected_nodes: Vec<usize>,
    argmin_data: ArgminData<Space::State, Space::Cost>,
}

pub enum ArgminImprovement<'a, S, C> {
    Unchanged,
    Improved(&'a ArgminData<S, C>),
}

impl<Space: DfaWithCost, M: NablaModel, P> NablaOptimizer<Space, M, P> {
    pub fn model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    pub fn trees(&self) -> &[SearchTree] {
        &self.trees
    }

    #[cfg(feature = "rayon")]
    pub fn par_new(
        space: Space,
        init_states: impl Fn() -> Space::State + Sync + Send,
        mut model: M,
        batch: usize,
        budget: &super::tree::graph_operations::ActionBudget,
    ) -> Self
    where
        Space: Sync,
        Space::State: Clone + Send + Sync,
        Space::Cost: Clone + Send + Sync,
        P: Send + crate::path::ActionPath + Ord,
    {
        use rayon::{
            iter::{
                IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator, ParallelIterator
            },
            slice::{ParallelSlice, ParallelSliceMut},
        };

        use crate::nabla::tree::state_weight::StateWeight;

        let roots: Vec<_> = (0..batch).into_par_iter().map(|_| init_states()).collect();
        let states = roots.clone();
        let costs: Vec<Space::Cost> = roots.as_slice().par_iter().map(|s| space.cost(s)).collect();
        let paths = (0..batch).into_par_iter().map(|_| P::new()).collect();
        let mut state_vecs = vec![0.; batch * Space::STATE_DIM];
        let mut action_bools = vec![false; batch * Space::ACTION_DIM];
        let mut num_actions = vec![0.; batch];
        (
            &states,
            state_vecs.par_chunks_exact_mut(Space::STATE_DIM),
            action_bools.par_chunks_exact_mut(Space::ACTION_DIM),
            num_actions.par_iter_mut(),
        )
            .into_par_iter()
            .for_each(|(s, s_host, a_host, num_actions)| {
                space.write_vecs(s, s_host, a_host, num_actions);
            });
        let mut v_host = vec![0.; batch * Space::ACTION_DIM];
        let mut p_host = vec![0.; batch * Space::ACTION_DIM];
        model.write_predictions(
            &state_vecs,
            &mut action_bools,
            &mut num_actions,
            &mut v_host,
            &mut p_host,
        );
        // h_theta_host.par_iter_mut().for_each(|x| *x = x.sqrt());
        let trees = (
            &states,
            &costs,
            v_host.par_chunks_exact(Space::ACTION_DIM),
            p_host.par_chunks_exact(Space::ACTION_DIM),
        )
            .into_par_iter()
            .map(|(s, c, h, p)| {
                let mut t = SearchTree::default();
                let c = space.evaluate(c);
                let weight = StateWeight::new(c);
                let root_id = t.add_node(weight);
                t.add_actions(root_id, &space, s, h, p, budget);
                t
            })
            .collect();
        // let transitions = (0..batch).into_par_iter().map(|_| Vec::new()).collect();
        let last_positions = vec![Default::default(); batch];
        // let action_weights = vec![0.; batch * Space::ACTION_DIM];
        let argmin_data: ArgminData<
            <Space as DfaWithCost>::State,
            <Space as DfaWithCost>::Cost,
        > = states
            .par_iter()
            .zip(costs.par_iter())
            .map(|(s, c)| (s, c, space.evaluate(c)))
            .min_by(|(_, _, e1), (_, _, e2)| e1.partial_cmp(e2).unwrap())
            .map(|(s, c, e)| ArgminData::new(s.clone(), c.clone(), e))
            .unwrap();
        let num_inspected_nodes = vec![0; batch];
        Self {
            space,
            roots,
            states,
            action_bools,
            num_actions,
            costs,
            paths,
            last_positions,
            state_vecs,
            v_host,
            p_host,
            // action_weights,
            trees,
            model,
            num_inspected_nodes,
            argmin_data,
        }
    }

    #[cfg(feature = "rayon")]
    pub fn par_roll_out_episodes(
        &mut self,
        n_as_tol: impl Fn(usize) -> u32 + Sync,
        budget: &super::tree::graph_operations::ActionBudget,
    ) -> ArgminImprovement<Space::State, Space::Cost>
    where
        Space: Sync,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
        P: Send
            + Sync
            + crate::path::ActionPath
            + crate::path::ActionPathFor<Space>
            + core::fmt::Debug,
    {
        use rayon::{
            iter::{IndexedParallelIterator, IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
            slice::{ParallelSlice, ParallelSliceMut},
        };

        let trees = &mut self.trees;
        let roots = &self.roots;
        let states = &mut self.states;
        let paths = &mut self.paths;
        let costs = &mut self.costs;
        let last_positions = &mut self.last_positions;
        self.state_vecs.fill(0.);
        self.action_bools.fill(false);
        self.num_actions.fill(0.);
        let state_vecs = self.state_vecs.par_chunks_exact_mut(Space::STATE_DIM);
        let action_bools = self.action_bools.par_chunks_exact_mut(Space::ACTION_DIM);
        let num_actions = self.num_actions.par_iter_mut();
        debug_assert!([
            roots.len(),
            states.len(),
            paths.len(),
            costs.len(),
            last_positions.len(),
            state_vecs.len(),
            action_bools.len(),
            num_actions.len(),
        ]
        .into_iter()
        .all(|l| l == trees.len()),);
        (
            trees,
            roots,
            states,
            paths,
            costs,
            last_positions,
            state_vecs,
            action_bools,
            num_actions,
        )
            .into_par_iter()
            .for_each(|(
                t,
                r,
                s,
                p,
                c,
                pos,
                v,
                a,
                num_actions,
            )| {
                t.roll_out_episodes(&self.space, r, s, c, p, pos, &n_as_tol);
                if !p.is_empty() {
                    self.space.write_vecs(s, v, a, num_actions);
                }
            });
        self.model
            .write_predictions(
                &self.state_vecs,
                &mut self.action_bools,
                &mut self.num_actions,
                &mut self.v_host,
                &mut self.p_host,
            );
        (
            &mut self.trees,
            &self.states,
            &self.last_positions,
            &self.paths,
            self.v_host.par_chunks_exact(Space::ACTION_DIM),
            self.p_host.par_chunks_exact(Space::ACTION_DIM),
        )
            .into_par_iter()
            .for_each(|(t, s, pos, p, h, prob)| {
                if !p.is_empty() {
                    t.add_actions(*pos, &self.space, s, h, prob, budget);
                }
            });
        self.par_update_argmmim_data()
    }

    #[cfg(feature = "rayon")]
    pub fn par_update_argmmim_data(&mut self) -> ArgminImprovement<Space::State, Space::Cost>
    where
        Space: Sync,
        P: Sync + crate::path::ActionPath,
        Space::State: Clone,
    {
        use rayon::iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator};

        let min_eval = self.argmin_data.eval;
        let n = (&mut self.num_inspected_nodes, &self.trees)
            .into_par_iter()
            .enumerate()
            .filter_map(|(tree_pos, (num, t))| {
                if *num < t.nodes().len() {
                    let n = t.nodes()[*num..]
                        .iter()
                        .enumerate()
                        .filter(|(_, n)| n.weight.c < min_eval)
                        .min_by(|(_, n1), (_, n2)| n1.weight.c.partial_cmp(&n2.weight.c).unwrap())
                        .map(|(i, n)| (tree_pos, i + *num, n));
                    // *num = 0;
                    *num = t.nodes().len();
                    n
                } else {
                    None
                }
            })
            .min_by(|(_, _, n1), (_, _, n2)| n1.weight.c.partial_cmp(&n2.weight.c).unwrap());
        match n {
            Some((tree_pos, node_pos, _n)) => {
                let tree = &self.trees[tree_pos];
                let ArgminData { state, cost, eval } = &mut self.argmin_data;
                state.clone_from(&self.roots[tree_pos]);
                let p: P = tree.find_path(NodeIndex::new(node_pos)).unwrap();
                p.actions_taken().for_each(|a| {
                    let a = self.space.action(a);
                    self.space.act(state, &a);
                });
                *cost = self.space.cost(state);
                *eval = self.space.evaluate(cost);
                ArgminImprovement::Improved(&self.argmin_data)
            }
            None => ArgminImprovement::Unchanged,
        }
    }

    #[cfg(feature = "rayon")]
    pub fn par_update_model(&mut self, n_obs_tol: u32) -> f32
    where
        Space: Sync,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
        P: Sync,
    {
        use rayon::{
            iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
            slice::ParallelSliceMut,
        };

        // sync `costs` with `roots`
        self.state_vecs.fill(0.);
        self.action_bools.fill(false);
        self.num_actions.fill(0.);
        let state_vecs = self.state_vecs.par_chunks_exact_mut(Space::STATE_DIM);
        let action_bools = self.action_bools.par_chunks_exact_mut(Space::ACTION_DIM);
        let num_actions = self.num_actions.par_iter_mut();
        (
            &self.roots,
            state_vecs,
            action_bools,
            num_actions,
        )
            .into_par_iter()
            .for_each(|(s, v, a, num_actions)| {
                self.space.write_vecs(s, v, a, num_actions);
            });

        // fill `h_theta_host`
        self.v_host.fill(-1.0);
        // panic!("{}", self.h_theta_host.len());
        // self.action_weights.fill(0.0);
        let h_theta_vecs = self.v_host.par_chunks_exact_mut(Space::ACTION_DIM);
        // let weight_vecs = self.action_weights.par_chunks_exact_mut(Space::ACTION_DIM);
        (&self.trees, h_theta_vecs)
            .into_par_iter()
            .for_each(|(t, h_theta)| {
                let foo = t.observations(&self.space, n_obs_tol);
                for f in foo {
                    h_theta[f.0] = f.1;
                }
                // t.write_observations(&self.space, h_theta, weights, n_obs_tol)
            });
        self.model
            .update_model(
                &self.state_vecs,
                &self.action_bools,
                &self.num_actions,
                &self.v_host,
                &self.p_host,
            )
    }

    #[cfg(feature = "rayon")]
    pub fn par_reset_trees<A, B>(
        &mut self,
        reset_policy: ResetPolicy<A, B>,
        // next_node: impl Fn(Vec<(NodeIndex, &super::tree::state_weight::StateWeight)>) -> NodeIndex
        //     + Sync,
        budget: &super::tree::graph_operations::ActionBudget,
    ) where
        Space: Sync,
        P: Send + Sync + crate::path::ActionPathFor<Space>,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
        A: Fn(&mut Space::State) + Sync,
        B: Fn(&mut Space::State) + Sync,
    {
        use rand::seq::IteratorRandom;
        use rayon::{
            iter::{IntoParallelIterator, IntoParallelRefMutIterator, ParallelIterator},
            slice::{ParallelSlice, ParallelSliceMut},
        };

        use crate::nabla::tree::state_weight::StateWeight;

        let Self {
            space: _,
            roots,
            states,
            action_bools,
            num_actions,
            costs,
            paths,
            last_positions,
            state_vecs,
            v_host,
            p_host,
            // action_weights: _,
            trees: _,
            model,
            num_inspected_nodes: _,
            argmin_data: _,
        } = self;
        last_positions.fill(Default::default());
        state_vecs.fill(0.);
        action_bools.fill(false);
        num_actions.fill(0.);
        let space = &self.space;
        let trees = &self.trees;
        let state_vecs = self.state_vecs.par_chunks_exact_mut(Space::STATE_DIM);
        let action_bools = self.action_bools.par_chunks_exact_mut(Space::ACTION_DIM);
        let num_actions = self.num_actions.par_iter_mut();
        // let next_node = &next_node;
        (
            trees, roots, states, costs, paths, // transitions,
            state_vecs, action_bools, num_actions,
        )
            .into_par_iter()
            .for_each(
                |(
                    t,
                    r,
                    s,
                    c,
                    p,
                    // trans,
                    v,
                    a,
                    num_actions,
                )| {
                    let nodes = t.nodes();
                    let (cost_threshold, improved) = nodes.first().map(|n| {
                        let w = &n.weight;
                        let (root_cost, best_cost) = (w.c, w.c_t_star);
                        if root_cost == best_cost {
                            (root_cost, false)
                        } else {
                            ((root_cost + reset_policy.greed * best_cost) / (1.0 + reset_policy.greed), true)
                        }
                    }).unwrap();
                    let viable_nodes = nodes.iter().enumerate().filter_map(|(i, n)| {
                        match n.weight.c <= cost_threshold {
                            true => Some(i),
                            false => None,
                        }
                    });
                    let mut rng = rand::thread_rng();
                    let next_node = viable_nodes.choose(&mut rng).map(|i| NodeIndex::new(i)).unwrap();
                    *p = t.find_path(next_node).unwrap();
                    for a in p.actions_taken() {
                        let a = space.action(a);
                        space.act(r, &a);
                    }
                    match improved {
                        true => (reset_policy.adjust_improved_root)(r),
                        false => (reset_policy.adjust_unimproved_root)(r),
                    }
                    s.clone_from(r);
                    *c = space.cost(r);
                    self.space.write_vecs(s, v, a, num_actions);
                    p.clear();
                },
            );
        v_host.fill(0.);
        model.write_predictions(
            &self.state_vecs,
            &mut self.action_bools,
            &mut self.num_actions,
            v_host,
            p_host,
        );
        let action_vecs = v_host.par_chunks_exact(Space::ACTION_DIM);
        let p_vecs = p_host.par_chunks_exact(Space::ACTION_DIM);
        (&mut self.trees, &self.roots, &self.costs, action_vecs, p_vecs)
            .into_par_iter()
            .for_each(|(t, r, c, h, p)| {
                t.clear();
                let c = space.evaluate(c);
                let weight = StateWeight::new(c);
                let root_id = t.add_node(weight);
                t.add_actions(root_id, space, r, h, p, budget);
            });
        self.num_inspected_nodes.fill(0);
    }
    
    pub fn argmin_data(&self) -> &ArgminData<Space::State, Space::Cost> {
        &self.argmin_data
    }
}

pub struct ResetPolicy<A, B> {
    pub greed: f32,
    pub adjust_improved_root: A,
    pub adjust_unimproved_root: B,
}
