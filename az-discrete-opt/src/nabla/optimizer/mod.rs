use core::num::NonZeroUsize;

use crate::log::ArgminData;

use super::{space::NablaStateActionSpace, tree::{SearchTree, Transition2}, model::NablaModel};

pub struct NablaOptimizer<Space: NablaStateActionSpace, M, P> {
    space: Space,
    roots: Vec<Space::State>,
    states: Vec<Space::State>,
    costs: Vec<Space::Cost>,
    paths: Vec<P>,
    transitions: Vec<Vec<Transition2>>,
    last_positions: Vec<Option<NonZeroUsize>>,
    state_vecs: Vec<f32>,
    h_theta_host: Vec<f32>,
    action_weights: Vec<f32>,
    trees: Vec<SearchTree<P>>,
    model: M,
    num_inspected_nodes: Vec<usize>,
    argmin_data: ArgminData<Space::State, Space::Cost>,
}

pub enum ArgminImprovement<'a, S, C> {
    Improved(&'a ArgminData<S, C>),
    Unchanged,
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
        P: Send + crate::path::ActionPath,
    {
        use rayon::{iter::{IntoParallelIterator, ParallelIterator, IntoParallelRefIterator, IndexedParallelIterator}, slice::{ParallelSlice, ParallelSliceMut}};

        let roots: Vec<_> = (0..batch).into_par_iter().map(|_| init_states()).collect();
        let states = roots.clone();
        let costs: Vec<Space::Cost> = roots.as_slice().par_iter().map(|s| space.cost(s)).collect();
        let paths = (0..batch).into_par_iter().map(|_| P::new()).collect();
        let mut state_vecs = vec![0.; batch * Space::STATE_DIM];
        (&states, state_vecs.par_chunks_exact_mut(Space::STATE_DIM)).into_par_iter().for_each(|(s, s_host)| {
            space.write_vec(s, s_host);
        });
        let mut h_theta_host = vec![0.; batch * Space::ACTION_DIM];
        model.write_predictions(&state_vecs, &mut h_theta_host);
        // h_theta_host.par_iter_mut().for_each(|x| *x = x.sqrt());
        let trees = 
            (&states, &costs, h_theta_host.par_chunks_exact(Space::ACTION_DIM))
            .into_par_iter()
            .map(|(s, c, h_theta)| {
                SearchTree::new(&space, s, c, h_theta)
            })
            .collect();
        let transitions = (0..batch).into_par_iter().map(|_| Vec::new()).collect();
        let last_positions = vec![None; batch];
        let action_weights = vec![0.; batch * Space::ACTION_DIM];
        let argmin_data =
            states.par_iter().zip(costs.par_iter())
            .map(|(s, c)| {
                (s, c, space.evaluate(c))
            })
            .min_by(|(_, _, e1), (_, _, e2)| {
            e1.partial_cmp(&e2).unwrap()
        }).map(|(s, c, e)| ArgminData::new(
            s.clone(),
            c.clone(),
            e,
        )).unwrap();
        let num_inspected_nodes = vec![0; batch];
        Self {
            space,
            roots,
            states,
            costs,
            paths,
            transitions,
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

    // #[cfg(feature = "rayon")]
    // pub fn par_argmin(&self) -> Option<(&Space::State, &Space::Cost, f32)>
    // where
    //     Space: Sync,
    //     Space::State: Send + Sync,
    //     Space::Cost: Send + Sync,
    // {
    //     use rayon::iter::{IntoParallelIterator, ParallelIterator};
    //     let Self {
    //         space,
    //         roots: _,
    //         states,
    //         costs,
    //         paths: _,
    //         transitions: _,
    //         last_positions: _,
    //         state_vecs: _,
    //         h_theta_host: _,
    //         action_weights: _,
    //         trees: _,
    //         model: _,

    //     } = self;
    //     (states, costs)
    //         .into_par_iter()
    //         .map(|(s, c)| (s, c, space.evaluate(c)))
    //         .min_by(|(_, _, e1), (_, _, e2)| e1.partial_cmp(e2).unwrap())
    // }

    #[cfg(feature = "rayon")]
    pub fn par_roll_out_episodes(
        &mut self,
        // action_pattern: impl Fn(usize) -> super::tree::node::SamplePattern + Sync,
        // policy: impl Fn(usize) -> super::tree::node::SearchPolicy + Sync,
    ) -> ArgminImprovement<Space::State, Space::Cost>
    where
        Space: Sync,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
        P: Ord + Clone + Send + Sync + crate::path::ActionPath + crate::path::ActionPathFor<Space>,
    {
        use rayon::{iter::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator}, slice::{ParallelSliceMut, ParallelSlice}};

        use crate::nabla::tree::node::StateNode2;

        let trees = &mut self.trees;
        let roots = &self.roots;
        let states = &mut self.states;
        let paths = &mut self.paths;
        let transitions = &mut self.transitions;
        let costs = &mut self.costs;
        let last_positions = &mut self.last_positions;
        let state_vecs = self.state_vecs.par_chunks_exact_mut(Space::STATE_DIM);
        debug_assert!(
            [
                roots.len(),
                states.len(),
                paths.len(),
                transitions.len(),
                costs.len(),
                last_positions.len(),
                state_vecs.len(),
            ]
            .into_iter()
            .all(|l| l == trees.len()),
        );
        (
            trees,
            roots,
            states,
            paths,
            transitions,
            costs,
            last_positions,
            state_vecs,
        ).into_par_iter().for_each(|(
            t,
            r,
            s,
            p,
            trans,
            c,
            pos,
            v,
        )| {
            t.roll_out_episodes(
                &self.space,
                r,
                s,
                c,
                p,
                trans,
                pos,
            );    
            if !trans.is_empty() {
                self.space.write_vec(s, v);
            }
        });
        self.model.write_predictions(&self.state_vecs, &mut self.h_theta_host);
        (
            &mut self.trees,
            &self.states,
            &self.costs,
            &self.transitions,
            self.h_theta_host.par_chunks_exact(Space::ACTION_DIM),
        ).into_par_iter().for_each(|(
            t,
            s,
            c,
            trans,
            h,
        )| {
            if !trans.is_empty() {
                let n = StateNode2::new(&self.space, s, c, h);
                t.push_node(n);
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
        use rayon::iter::{IntoParallelIterator, ParallelIterator, IndexedParallelIterator};

        let min_eval = self.argmin_data.eval;
        let n = (&mut self.num_inspected_nodes, &self.trees)
            .into_par_iter()
            .enumerate()
            .filter_map(|(tree_pos, (num, t))| {
                if *num < t.nodes().len() {
                    let n =
                        t.nodes()[*num..]
                        .iter()
                        .enumerate()
                        .filter(|(_, n)| {
                            n.c < min_eval
                        }).min_by(|(_, n1), (_, n2)| {
                            n1.c.partial_cmp(&n2.c).unwrap()
                        })
                        .map(|(i, n)| (tree_pos, i + *num, n));
                    // *num = 0;
                    *num = t.nodes().len();
                    n
                } else {
                    None
                }
            })
            .min_by(|(_, _, n1), (_, _, n2)| {
                n1.c.partial_cmp(&n2.c).unwrap()
            });
        match n {
            Some((tree_pos, node_pos, n)) => {
                let tree = &self.trees[tree_pos];
                let ArgminData { state, cost, eval } = &mut self.argmin_data;
                state.clone_from(&self.roots[tree_pos]);
                let p: Option<&P> = tree.positions().iter().find_map(|(p, pos)| {
                    if pos.get() == node_pos {
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
                } else {
                    todo!()
                }
                *cost = self.space.cost(state);
                *eval = self.space.evaluate(cost);
                ArgminImprovement::Improved(&self.argmin_data)
            },
            None => ArgminImprovement::Unchanged,
        }
    }

    #[cfg(feature = "rayon")]
    pub fn par_update_model(
        &mut self,
        // action_weights: impl Fn(u32) -> f32 + Sync,
        // state_weights: impl Fn(&Space::State) -> f32 + Sync,
    ) -> f32
    where
        Space: Sync,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
        P: Sync,
    {
        use rayon::{iter::{IntoParallelIterator, ParallelIterator}, slice::ParallelSliceMut};

        // sync `costs` with `roots`
        let state_vecs = self.state_vecs.par_chunks_exact_mut(Space::STATE_DIM);
        (
            &self.roots,
            state_vecs,
        ).into_par_iter().for_each(|(s, v)| {
            self.space.write_vec(s, v);
        });

        // fill `h_theta_host`
        self.h_theta_host.fill(0.0);
        self.action_weights.fill(0.0);
        let h_theta_vecs = self.h_theta_host.par_chunks_exact_mut(Space::ACTION_DIM);
        let weight_vecs = self.action_weights.par_chunks_exact_mut(Space::ACTION_DIM);
        (&self.trees, h_theta_vecs, weight_vecs)
            .into_par_iter()
            .for_each(|(t, h_theta, weights)|
                t.write_observations(&self.space, h_theta, weights)
            );
        self.model.update_model(&self.state_vecs, &self.h_theta_host, &self.action_weights)
    }

    #[cfg(feature = "rayon")]
    pub fn reset_trees(
        &mut self,
        modify_root: impl Fn(&Space, &mut Space::State, Vec<(Option<&P>, f32, f32)>) + Sync,
    )
    where
        Space: Sync,
        P: Ord + Send + Sync + crate::path::ActionPathFor<Space>,
        Space::State: Clone + Send + Sync,
        Space::Cost: Send + Sync,
    {
        use rayon::{iter::{ParallelIterator, IntoParallelIterator}, slice::{ParallelSliceMut, ParallelSlice}};

        let Self {
            space: _,
            roots,
            states,
            costs,
            paths,
            transitions,
            last_positions,
            state_vecs,
            h_theta_host,
            action_weights: _,
            trees: _,
            model,
            num_inspected_nodes: _,
            argmin_data: _,
        } = self;
        last_positions.fill(None);
        state_vecs.fill(0.);
        let space = &self.space;
        let trees = &self.trees;
        let state_vecs = self.state_vecs.par_chunks_exact_mut(Space::STATE_DIM);
        let modify_root = &modify_root;
        (
            trees,
            roots,
            states,
            costs,
            paths,
            transitions,
            state_vecs,
        ).into_par_iter().for_each(|(
            t,
            r,
            s,
            c,
            p,
            trans,
            v,
        )| {
            let n = t.node_data();
            modify_root(space, r, n);
            s.clone_from(r);
            *c = space.cost(r);
            self.space.write_vec(s, v);
            p.clear();
            trans.clear();
        });
        h_theta_host.fill(0.);
        model.write_predictions(&self.state_vecs, h_theta_host);
        let action_vecs = h_theta_host.par_chunks_exact(Space::ACTION_DIM);
        (
            &mut self.trees,
            &self.roots,
            &self.costs,
            action_vecs,
        ).into_par_iter().for_each(|(
            t,
            r,
            c,
            h,
        )| {
            *t = SearchTree::new(&self.space, r, c, h);
        });
        self.num_inspected_nodes.fill(0);
    }

    #[cfg(feature = "rayon")]
    fn par_reset_trees(&mut self)
    where
        Space: Sync,
        Space::State: Sync,
        Space::Cost: Send + Sync,
        P: Send + crate::path::ActionPath,
    {
        use rayon::{iter::{IntoParallelIterator, ParallelIterator}, slice::ParallelSliceMut};

        (&self.roots, &mut self.costs).into_par_iter().for_each(|(s, c)| {
            *c = self.space.cost(s);
        });
        let state_vecs = self.state_vecs.par_chunks_exact_mut(Space::STATE_DIM);
        (&self.states, state_vecs).into_par_iter().for_each(|(s, s_host)| {
            self.space.write_vec(s, s_host);
        });
        self.model.write_predictions(&self.state_vecs, &mut self.h_theta_host);
        let h_theta_vecs = self.h_theta_host.par_chunks_exact_mut(Space::ACTION_DIM);
        (&mut self.trees, &self.roots, &self.costs, h_theta_vecs).into_par_iter().for_each(|(t, s, c, h_theta)| {
            // todo! ?perf
            // dbg!();
            *t = SearchTree::new(&self.space, s, c, h_theta);
        });
        self.last_positions.fill(None);
        (&mut self.paths, &mut self.transitions)
            .into_par_iter()
            .for_each(|(p, t)| {
                p.clear();
                t.clear();
            });
        // todo!("anything else?")
    }

    pub fn argmin_data(&self) -> &ArgminData<Space::State, Space::Cost> {
        &self.argmin_data
    }
}
