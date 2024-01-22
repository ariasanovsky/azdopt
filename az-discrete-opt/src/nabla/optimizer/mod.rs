use petgraph::stable_graph::NodeIndex;

use crate::log::ArgminData;

use super::{model::NablaModel, space::NablaStateActionSpace, tree::SearchTree};

pub struct NablaOptimizer<Space: NablaStateActionSpace, M, P> {
    space: Space,
    roots: Vec<Space::State>,
    states: Vec<Space::State>,
    costs: Vec<Space::Cost>,
    paths: Vec<P>,
    // transitions: Vec<Vec<Transition>>,
    last_positions: Vec<NodeIndex>,
    state_vecs: Vec<f32>,
    h_theta_host: Vec<f32>,
    action_weights: Vec<f32>,
    trees: Vec<SearchTree<P>>,
    model: M,
    num_inspected_nodes: Vec<usize>,
    argmin_data: ArgminData<Space::State, Space::Cost>,
}

pub enum ArgminImprovement<'a, S, C> {
    Unchanged,
    Improved(&'a ArgminData<S, C>),
}

impl<Space: NablaStateActionSpace, M: NablaModel, P> NablaOptimizer<Space, M, P> {
    pub fn get_model_mut(&mut self) -> &mut M {
        &mut self.model
    }

    pub fn get_trees(&self) -> &[SearchTree<P>] {
        &self.trees
    }

    #[cfg(feature = "rayon")]
    pub fn par_new(
        space: Space,
        init_states: impl Fn() -> Space::State + Sync + Send,
        mut model: M,
        batch: usize,
    ) -> Self
    where
        Space: Sync,
        Space::State: Clone + Send + Sync,
        Space::Cost: Clone + Send + Sync,
        P: Send + crate::path::ActionPath + Ord,
    {
        use rayon::{
            iter::{
                IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator,
                ParallelIterator,
            },
            slice::{ParallelSlice, ParallelSliceMut},
        };

        use crate::nabla::tree::state_weight::StateWeight;

        let roots: Vec<_> = (0..batch).into_par_iter().map(|_| init_states()).collect();
        let states = roots.clone();
        let costs: Vec<Space::Cost> = roots.as_slice().par_iter().map(|s| space.cost(s)).collect();
        let paths = (0..batch).into_par_iter().map(|_| P::new()).collect();
        let mut state_vecs = vec![0.; batch * Space::STATE_DIM];
        (&states, state_vecs.par_chunks_exact_mut(Space::STATE_DIM))
            .into_par_iter()
            .for_each(|(s, s_host)| {
                space.write_vec(s, s_host);
            });
        let mut h_theta_host = vec![0.; batch * Space::ACTION_DIM];
        model.write_predictions(&state_vecs, &mut h_theta_host);
        // h_theta_host.par_iter_mut().for_each(|x| *x = x.sqrt());
        let trees = (
            &states,
            &costs,
            h_theta_host.par_chunks_exact(Space::ACTION_DIM),
        )
            .into_par_iter()
            .map(|(s, c, h)| {
                let mut t = SearchTree::default();
                let c = space.evaluate(c);
                let weight = StateWeight::new(c);
                let root_id = t.add_node(P::new(), weight);
                t.add_actions(root_id, &space, s, h);
                t
            })
            .collect();
        // let transitions = (0..batch).into_par_iter().map(|_| Vec::new()).collect();
        let last_positions = vec![Default::default(); batch];
        let action_weights = vec![0.; batch * Space::ACTION_DIM];
        let argmin_data: ArgminData<
            <Space as NablaStateActionSpace>::State,
            <Space as NablaStateActionSpace>::Cost,
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
            costs,
            paths,
            last_positions,
            state_vecs,
            h_theta_host,
            action_weights,
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
    ) -> ArgminImprovement<Space::State, Space::Cost>
    where
        Space: Sync,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
        P: Ord
            + Clone
            + Send
            + Sync
            + crate::path::ActionPath
            + crate::path::ActionPathFor<Space>
            + core::fmt::Debug,
    {
        use rayon::{
            iter::{IndexedParallelIterator, IntoParallelIterator, ParallelIterator},
            slice::{ParallelSlice, ParallelSliceMut},
        };

        let trees = &mut self.trees;
        let roots = &self.roots;
        let states = &mut self.states;
        let paths = &mut self.paths;
        let costs = &mut self.costs;
        let last_positions = &mut self.last_positions;
        let state_vecs = self.state_vecs.par_chunks_exact_mut(Space::STATE_DIM);
        debug_assert!([
            roots.len(),
            states.len(),
            paths.len(),
            costs.len(),
            last_positions.len(),
            state_vecs.len(),
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
        )
            .into_par_iter()
            .for_each(|(t, r, s, p, c, pos, v)| {
                t.roll_out_episodes(&self.space, r, s, c, p, pos, &n_as_tol);
                if !p.is_empty() {
                    self.space.write_vec(s, v);
                }
            });
        self.model
            .write_predictions(&self.state_vecs, &mut self.h_theta_host);
        (
            &mut self.trees,
            &self.states,
            &self.last_positions,
            &self.paths,
            self.h_theta_host.par_chunks_exact(Space::ACTION_DIM),
        )
            .into_par_iter()
            .for_each(|(t, s, pos, p, h)| {
                if !p.is_empty() {
                    t.add_actions(*pos, &self.space, s, h);
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
                let p = tree.positions().iter().find_map(|(p, pos)| {
                    if pos.index() == node_pos {
                        Some(p)
                    } else {
                        None
                    }
                });
                if let Some(p) = p {
                    p.actions_taken().for_each(|a| {
                        let a = self.space.action(a);
                        self.space.act(state, &a);
                    });
                }
                *cost = self.space.cost(state);
                *eval = self.space.evaluate(cost);
                ArgminImprovement::Improved(&self.argmin_data)
            }
            None => ArgminImprovement::Unchanged,
        }
    }

    #[cfg(feature = "rayon")]
    pub fn par_update_model(&mut self, n_as_tol: u32) -> f32
    where
        Space: Sync,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
        P: Sync,
    {
        use rayon::{
            iter::{IntoParallelIterator, ParallelIterator},
            slice::ParallelSliceMut,
        };

        // sync `costs` with `roots`
        let state_vecs = self.state_vecs.par_chunks_exact_mut(Space::STATE_DIM);
        (&self.roots, state_vecs)
            .into_par_iter()
            .for_each(|(s, v)| {
                self.space.write_vec(s, v);
            });

        // fill `h_theta_host`
        self.h_theta_host.fill(0.0);
        self.action_weights.fill(0.0);
        let h_theta_vecs = self.h_theta_host.par_chunks_exact_mut(Space::ACTION_DIM);
        let weight_vecs = self.action_weights.par_chunks_exact_mut(Space::ACTION_DIM);
        (&self.trees, h_theta_vecs, weight_vecs)
            .into_par_iter()
            .for_each(|(t, h_theta, weights)| {
                t.write_observations(&self.space, h_theta, weights, n_as_tol)
            });
        self.model
            .update_model(&self.state_vecs, &self.h_theta_host, &self.action_weights)
    }

    #[cfg(feature = "rayon")]
    pub fn par_reset_trees(
        &mut self,
        modify_root: impl Fn(&Space, &mut Space::State, Vec<(&P, &super::tree::state_weight::StateWeight)>)
            + Sync,
    ) where
        Space: Sync,
        P: Ord + Send + Sync + crate::path::ActionPathFor<Space>,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
    {
        use rayon::{
            iter::{IntoParallelIterator, ParallelIterator},
            slice::{ParallelSlice, ParallelSliceMut},
        };

        use crate::nabla::tree::state_weight::StateWeight;

        let Self {
            space: _,
            roots,
            states,
            costs,
            paths,
            // transitions,
            last_positions,
            state_vecs,
            h_theta_host,
            action_weights: _,
            trees: _,
            model,
            num_inspected_nodes: _,
            argmin_data: _,
        } = self;
        last_positions.fill(Default::default());
        state_vecs.fill(0.);
        let space = &self.space;
        let trees = &self.trees;
        let state_vecs = self.state_vecs.par_chunks_exact_mut(Space::STATE_DIM);
        let modify_root = &modify_root;
        (
            trees, roots, states, costs, paths, // transitions,
            state_vecs,
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
                )| {
                    let n = t.node_data();
                    modify_root(space, r, n);
                    s.clone_from(r);
                    *c = space.cost(r);
                    self.space.write_vec(s, v);
                    p.clear();
                    // trans.clear();
                },
            );
        h_theta_host.fill(0.);
        model.write_predictions(&self.state_vecs, h_theta_host);
        let action_vecs = h_theta_host.par_chunks_exact(Space::ACTION_DIM);
        (&mut self.trees, &self.roots, &self.costs, action_vecs)
            .into_par_iter()
            .for_each(|(t, r, c, h)| {
                t.clear();
                let c = space.evaluate(c);
                let weight = StateWeight::new(c);
                let root_id = t.add_node(P::new(), weight);
                t.add_actions(root_id, space, r, h);
            });
        self.num_inspected_nodes.fill(0);
    }
    pub fn argmin_data(&self) -> &ArgminData<Space::State, Space::Cost> {
        &self.argmin_data
    }
}
