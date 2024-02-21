use crate::path::{ActionPath, ActionPathFor};

use self::{
    arc_weight::{ActionPrediction, ActionWeight},
    state_weight::StateWeight,
};

use super::space::NablaStateActionSpace;

use petgraph::{
    graph::DiGraph,
    visit::{EdgeRef, IntoNodeReferences},
};

pub(crate) use petgraph::stable_graph::{EdgeIndex, NodeIndex};

pub mod arc_weight;
pub(crate) mod empty_transitions;
pub(crate) mod find_path;
pub mod graph_operations;
#[cfg(feature = "graphviz")]
pub mod graphviz;
pub mod next_action;
pub mod node;
pub mod state_weight;

pub struct SearchTree {
    tree: DiGraph<StateWeight, ActionWeight>,
    predictions: Vec<ActionPrediction>,
}

impl Default for SearchTree {
    fn default() -> Self {
        Self {
            tree: Default::default(),
            predictions: Default::default(),
        }
    }
}

impl SearchTree {
    pub fn clear(&mut self) {
        self.tree.clear();
        self.predictions.clear();
    }

    pub fn sizes<P>(&self) -> Vec<(usize, usize)>
    where
        P: ActionPath,
    {
        // todo!();
        let mut sizes = vec![(0, 0)];
        // println!("{}", self.positions::<P>().len());
        // for (p, i) in self.positions::<P>().iter() {
        //     let len = p.len();
        //     if len >= sizes.len() {
        //         sizes.resize(len + 1, (0, 0));
        //     }
        //     sizes[len].0 += 1;
        //     if self.tree[*i].is_active() {
        //         sizes[len].1 += 1;
        //     }
        // }
        sizes
    }

    pub(crate) fn _exhausted_nodes(&self) -> Vec<usize> {
        self.tree
            .node_weights()
            .enumerate()
            .filter_map(|(i, w)| if !w.is_active() { Some(i) } else { None })
            .collect()
    }

    pub(crate) fn _print_neighborhoods(&self) {
        for (node, weight) in self.tree.node_references() {
            print!(
                "{node:?}\t({})\t({:?})  \t{{",
                weight.n_t(),
                weight.actions,
            );
            for e in self
                .tree
                .neighbors_directed(node, petgraph::Direction::Outgoing)
            {
                print!("{e:?}, ");
            }
            println!("}}")
        }
        for edge in self.tree.edge_references() {
            let weight = edge.weight();
            let prediction = &self.predictions[weight.prediction_pos];
            let id = edge.id();
            let source = edge.source();
            let target = edge.target();
            println!("{id:?}\t{source:?}\t->\t{target:?}\t{weight:?}\t{prediction:?}");
        }
    }

    pub(crate) fn _permitted_actions(&self, node: NodeIndex) -> Vec<usize> {
        let action_range = &self.tree[node].actions;
        let action_range = (action_range.start as usize)..(action_range.end as usize);
        self.predictions[action_range]
            .iter()
            .map(|p| p.a_id)
            .collect()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn roll_out_episodes<Space, P>(
        &mut self,
        space: &Space,
        root: &Space::State,
        state: &mut Space::State,
        cost: &mut Space::Cost,
        path: &mut P,
        state_pos: &mut NodeIndex,
        n_as_tol: impl Fn(usize) -> u32,
    ) where
        Space: NablaStateActionSpace,
        Space::State: Clone,
        P: ActionPath + ActionPathFor<Space>, // + core::fmt::Debug,
    {
        loop {
            // println!("path: {path:?}");
            // let possible_actions = space.action_data(state).map(|(a, _)| a).collect::<Vec<_>>();
            // eprintln!("{state_pos:?}");
            // eprintln!("possible_actions: {possible_actions:?}");
            // eprintln!("permited_actions: {:?}", self._permitted_actions(*state_pos));
            // eprintln!("exhausted_nodes: {:?}", self._exhausted_nodes());
            // self._print_neighborhoods();

            use next_action::NextAction;
            let next_action = if path.is_empty() {
                self.sample_actions(*state_pos, n_as_tol(path.len()))
            } else {
                self.next_optimal_action(*state_pos, n_as_tol(path.len()))
            };
            match next_action {
                Some(NextAction::Visited(arc_index)) => {
                    // eprintln!(
                    //     "\trevisiting arc {arc_index:?} (from {:?} to {:?})",
                    //     self.tree.edge_endpoints(arc_index).unwrap().0,
                    //     self.tree.edge_endpoints(arc_index).unwrap().1,
                    // );

                    let prediction_pos = self.tree[arc_index].prediction_pos;
                    let action_id = self.predictions[prediction_pos].a_id;
                    unsafe { path.push_unchecked(action_id) };
                    let action = space.action(action_id);
                    space.act(state, &action);
                    *state_pos = self.tree.edge_endpoints(arc_index).unwrap().1;
                    // if !self.tree[*state_pos].n_t.is_active() {
                    //     eprintln!("\tterminal");
                    //     // todo!();
                    // } else {
                    //     eprintln!("\tnot terminal");
                    //     // todo!();
                    // }
                }
                Some(NextAction::Unvisited(prediction_pos)) => {
                    let range = &self.tree[*state_pos].actions.clone();
                    let range = (range.start as usize)..(range.end as usize);
                    debug_assert!(range.contains(&prediction_pos));
                    let prediction = &self.predictions[prediction_pos];
                    let action_id = prediction.a_id;
                    // eprintln!(
                    //     "\tfirst visit to {action_id} (pos = {prediction_pos}) from {state_pos:?}",
                    // );
                    unsafe { path.push_unchecked(action_id) };
                    let next_pos: Option<NodeIndex> = self.find_node(path);
                    // todo!(); //self.positions::<P>().get(path);
                    match next_pos {
                        Some(next_pos) => {
                            let arc_index = self.add_arc(*state_pos, next_pos, prediction_pos);
                            // println!("\trediscovered node, resetting!\n");
                            self.cascade_old_node(arc_index);
                            state.clone_from(root);
                            path.clear();
                            *state_pos = Default::default();
                        }
                        None => {
                            let action = space.action(action_id);
                            space.act(state, &action);
                            *cost = space.cost(state);
                            let c_as = space.evaluate(cost);
                            // let c_as_star = c_as;
                            let weight = StateWeight::new(c_as);
                            // eprint!("\tnew node_weight {weight:?}");
                            let next_pos = self.add_node(weight);
                            // eprintln!(" on {next_pos:?}");
                            // let c_s = self.tree.node_weight(*state_pos).unwrap().c;
                            // let g_t_sa = c_s - c_as_star;
                            // let h_t_sa = space.h_sa(c_s, c_as, c_as_star);
                            // let arc_weight = ActionWeight {
                            //     // g_t_sa,
                            //     // h_t_sa: FiniteOrExhausted(h_t_sa),
                            //     prediction_pos,
                            // };
                            let arc_index = self.add_arc(*state_pos, next_pos, prediction_pos);
                            match space.is_terminal(state) {
                                true => {
                                    // self.tree[next_pos].mark_exhausted();
                                    // eprintln!("\texhausted_nodes: {:?}", self._exhausted_nodes());
                                    self.cascade_new_terminal(arc_index);
                                    // eprintln!("\t->               {:?}", self._exhausted_nodes());
                                    // todo!("terminal, so cascade updates and start over!");
                                    state.clone_from(root);
                                    path.clear();
                                    *state_pos = Default::default();
                                    // eprintln!("\treset!\n");
                                }
                                false => {
                                    // eprintln!("\trequires prediction!");
                                    *state_pos = next_pos;
                                    return;
                                }
                            }
                        }
                    }
                }
                None => match path.is_empty() {
                    true => {
                        debug_assert_eq!(state_pos.index(), 0);
                        return;
                    }
                    false => {
                        todo!();
                        // debug_assert!(self.positions::<P>().contains_key(path));
                        unreachable!("asdkfnasd;fkna");
                    }
                },
            }
        }
    }

    pub(crate) fn write_observations<Space: NablaStateActionSpace>(
        &self,
        space: &Space,
        observations: &mut [f32],
        weights: &mut [f32],
        n_t_as_tol: u32,
    ) {
        let c_s = self.tree[NodeIndex::default()].c;
        for e in self
            .tree
            .edges_directed(NodeIndex::default(), petgraph::Direction::Outgoing)
        {
            let child_weight = self.tree.node_weight(e.target()).unwrap();
            if !child_weight.is_active() || child_weight.n_t() >= n_t_as_tol {
                let c_as = child_weight.c;
                let c_as_star = child_weight.c_t_star;
                let h_t_sa = space.h_sa(c_s, c_as, c_as_star);
                let prediction_pos = e.weight().prediction_pos;
                let action_id = self.predictions[prediction_pos].a_id;
                observations[action_id] = h_t_sa;
                weights[action_id] = 1.0;
            }
        }
        // .filter_map(|e| {
        //     let child_weight = self.tree.node_weight(e.target()).unwrap();
        //     child_weight.n_t.map(|n_as| {
        //         let arc_weight = self.tree.edge_weight(e.id()).unwrap();
        //         let prediction = &self.predictions[arc_weight.prediction_pos];
        //         let h_theta_sa = prediction.h_theta_sa;
        //         let n_as = n_as.get() - 1;
        //         debug_assert_ne!(n_as, 0);
        //         let concern = (h_t_sa - h_theta_sa).powi(2) / n_as as f32;
        //         (e.id(), concern)
        //     })
        // });

        // for e in self.tree.edges_directed(NodeIndex::default(), petgraph::Direction::Outgoing) {
        //     todo!()
        // }
        // todo!();
        // let root_node = &self.nodes[0];
        // let c_s = root_node.c;
        // dbg!(c_s);
        // root_node
        //     .actions
        //     .iter()
        //     .filter_map(|a| {
        //         a.next_position().map(|node_as| {
        //             let node_as = &self.nodes[node_as.get()];
        //             (a.action(), node_as.c, node_as.c_star)
        //         })
        //     })
        //     .for_each(|(a, c_as, c_as_star)| {
        //         let h = space.h_sa(c_s, c_as, c_as_star);
        //         // dbg!(h, a, c_as, c_as_star);
        //         observations[a] = h;
        //         weights[a] = 1.0;
        //     });
    }

    // pub(crate) fn node_data(&self) -> Vec<(NodeIndex, &StateWeight)> {
    //     todo!()
    //     // self.positions
    //     //     .iter()
    //     //     .map(|(p, i)| (p, &self.tree[*i]))
    //     //     .collect()
    // }

    // pub(crate) fn positions<P>(&self) -> &BTreeMap<P, NodeIndex> {
    //     todo!()
    //     // &self.positions
    // }

    pub(crate) fn nodes(&self) -> &[petgraph::graph::Node<StateWeight>] {
        self.tree.raw_nodes()
    }
}
