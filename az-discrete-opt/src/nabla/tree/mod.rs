use std::collections::BTreeMap;

use crate::path::{ActionPath, ActionPathFor};

use self::{state_weight::StateWeight, arc_weight::{ActionWeight, ActionPrediction}};

use super::space::NablaStateActionSpace;

use petgraph::{graph::DiGraph, visit::{IntoNodeReferences, EdgeRef}};

pub(crate) use petgraph::stable_graph::{EdgeIndex, NodeIndex};

pub mod empty_transitions;
#[cfg(feature = "graphviz")]
pub mod graphviz;
pub mod next_action;
pub mod node;
pub mod graph_operations;
pub mod state_weight;
pub mod arc_weight;

pub struct SearchTree<P> {
    positions: BTreeMap<P, NodeIndex>,
    tree: DiGraph<StateWeight, ActionWeight>,
    predictions: Vec<ActionPrediction>,
}

impl<P> Default for SearchTree<P> {
    fn default() -> Self {
        Self { positions: Default::default(), tree: Default::default(), predictions: Default::default() }
    }
}

impl<P> SearchTree<P> {
    pub fn clear(&mut self) {
        self.positions.clear();
        self.tree.clear();
        self.predictions.clear();
    }

    pub fn sizes(&self) -> impl Iterator<Item = (usize, usize)> + '_
    where
        P: ActionPath,
    {
        let mut sizes = vec![(0, 0)];
        println!("{}", self.positions.len());
        for (p, i) in self.positions.iter() {
            let len = p.len();
            if len >= sizes.len() {
                sizes.resize(len + 1, (0, 0));
            }
            sizes[len].0 += 1;
            if self.tree[*i].n_t.is_none() {
                sizes[len].1 += 1;
            }
        }
        sizes.into_iter()
    }

    pub(crate) fn _exhausted_nodes(&self) -> Vec<usize> {
        self.tree.node_weights().enumerate().filter_map(|(i, w)| {
            if w.n_t.is_none() {
                Some(i)
            } else {
                None
            }
        }).collect()
    }

    pub(crate) fn _print_neighborhoods(&self) {
        for (node, weight) in self.tree.node_references() {
            print!(
                "{node:?}\t({})\t({:?})  \t{{",
                weight.n_t.map_or(0, std::num::NonZeroU32::get),
                weight.actions,
            );
            for e in self.tree.neighbors_directed(node, petgraph::Direction::Outgoing) {
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
        let action_range = self.tree[node].actions.clone();
        self.predictions[action_range].iter().map(|p| p.a_id).collect()
    }

    // #[allow(clippy::too_many_arguments)]
    pub fn roll_out_episodes<Space>(
        &mut self,
        space: &Space,
        root: &Space::State,
        state: &mut Space::State,
        cost: &mut Space::Cost,
        path: &mut P,
        state_pos: &mut NodeIndex,
    ) where
        Space: NablaStateActionSpace,
        Space::State: Clone,
        P: Ord + ActionPath + ActionPathFor<Space> + Clone,// + core::fmt::Debug,
    {
        loop {
            // println!("path: {path:?}");
            // let possible_actions = space.action_data(state).map(|(a, _)| a).collect::<Vec<_>>();
            // println!("{state_pos:?}");
            // println!("possible_actions: {possible_actions:?}");
            // println!("permited_actions: {:?}", self._permitted_actions(*state_pos));
            // println!("exhausted_nodes: {:?}", self._exhausted_nodes());
            // self._print_neighborhoods();


            use next_action::NextAction;
            let next_action = self.next_action(*state_pos);
            match next_action {
                Some(NextAction::Visited(arc_index)) => {
                    // println!(
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
                    // if self.tree[*state_pos].n_t.is_none() {
                    //     println!("\tterminal");
                    //     // todo!();
                    // } else {
                    //     println!("\tnot terminal");
                    //     // todo!();
                    // }
                },
                Some(NextAction::Unvisited(prediction_pos)) => {
                    let range = self.tree[*state_pos].actions.clone();
                    debug_assert!(range.contains(&prediction_pos));
                    let prediction = &self.predictions[prediction_pos];
                    let action_id = prediction.a_id;
                    // println!(
                    //     "\tfirst visit to {action_id} (pos = {prediction_pos}) from {state_pos:?}",
                    // );
                    unsafe { path.push_unchecked(action_id) };
                    let next_pos = self.positions.get(path);
                    match next_pos {
                        Some(next_pos) => {
                            let arc_index = self.add_arc(*state_pos, *next_pos, prediction_pos);
                            // println!("\trediscovered node, resetting!\n");
                            self.cascade_updates(arc_index);
                            state.clone_from(root);
                            path.clear();
                            *state_pos = Default::default();
                        },
                        None => {
                            let action = space.action(action_id);
                            space.act(state, &action);
                            *cost = space.cost(state);
                            let c_as = space.evaluate(cost);
                            // let c_as_star = c_as;
                            let weight = StateWeight::new(c_as);
                            // print!("\tnew node_weight {weight:?}");
                            let next_pos = self.add_node(path.clone(), weight);
                            // println!(" on {next_pos:?}");
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
                                    self.tree[next_pos].assert_exhausted();
                                    // println!("\texhausted_nodes: {:?}", self._exhausted_nodes());
                                    self.cascade_updates(arc_index);
                                    // println!("\t->               {:?}", self._exhausted_nodes());
                                    // todo!("terminal, so cascade updates and start over!");
                                    state.clone_from(root);
                                    path.clear();
                                    *state_pos = Default::default();
                                    // println!("\treset!\n");
                                },
                                false => {
                                    // println!("\trequires prediction!");
                                    *state_pos = next_pos;
                                    return;
                                },
                            }
                        },
                    }
                },
                None => {
                    match path.is_empty() {
                        true => {
                            debug_assert_eq!(state_pos.index(), 0);
                            return;
                        },
                        false => {
                            debug_assert!(self.positions.contains_key(path));
                            unreachable!("asdkfnasd;fkna");
                        },
                    }
                },
            }
        }
    }

    // pub(crate) fn push_node(&mut self, node: StateNode)
    // where
    //     P: Ord + ActionPath,
    // {
    //     todo!();
    //     // self.nodes.push(node);
    // }

    pub(crate) fn write_observations<Space: NablaStateActionSpace>(
        &self,
        space: &Space,
        observations: &mut [f32],
        weights: &mut [f32],
    ) {
        observations.fill(0.0);
        weights.fill(0.0);
        let c_s = self.tree[NodeIndex::default()].c;
        for e in self.tree.edges_directed(NodeIndex::default(), petgraph::Direction::Outgoing) {
                let child_weight = self.tree.node_weight(e.target()).unwrap();
                if !child_weight.n_t.is_some_and(|n_t| {
                    n_t.get() < 25
                }) {
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

    pub(crate) fn node_data(&self) -> Vec<(&P, &StateWeight)> {
        self.positions
            .iter()
            .map(|(p, i)| (p, &self.tree[*i]))
            .collect()
    }

    pub(crate) fn positions(&self) -> &BTreeMap<P, NodeIndex> {
        &self.positions
    }

    pub(crate) fn nodes(&self) -> &[petgraph::graph::Node<StateWeight>] {
        self.tree.raw_nodes()
    }
}
