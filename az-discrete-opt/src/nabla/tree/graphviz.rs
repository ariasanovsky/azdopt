use dot_generator::*;
use dot_structures::*;
use graphviz_rust::{cmd::Format, exec, printer::PrinterContext};
use petgraph::visit::IntoNodeReferences;

use super::{state_weight::StateWeight, SearchTree};

impl SearchTree {
    pub fn graphviz(&self) -> Vec<u8> {
        let foo = |u: crate::nabla::tree::NodeIndex, n: &StateWeight| -> String {
            format!("s{}n{}x{}", u.index(), n.n_t(), n.exhausted_children)
        };

        let mut g = graph!(id!("search_tree"));
        for (u, n) in self.tree.node_references() {
            // let u_id = u.index();
            let foo = foo(u, n);
            let node = match n.is_active() {
                false => node!(
                    foo;
                    attr!("shape", "doublecircle")
                ),
                true => node!(foo),
            };
            g.add_stmt(Stmt::Node(node));
        }
        for u in self.tree.node_indices() {
            // let u_id = u.index();
            let u_label = foo(u, &self.tree[u]);
            for v in self
                .tree
                .neighbors_directed(u, petgraph::Direction::Outgoing)
            {
                // let v_id = v.index();
                let v_label = foo(v, &self.tree[v]);
                let edge = match self.tree[v].is_active() {
                    false => edge!(
                        node_id!(u_label) => node_id!(v_label)
                    ),
                    true => edge!(
                        node_id!(u_label) => node_id!(v_label);
                        attr!("dir", "forward")
                    ),
                };
                let e = Stmt::Edge(edge);
                g.add_stmt(e);
            }
        }
        // for (u, n) in self.nodes.iter().enumerate() {
        //     let node = match n.is_exhausted() {
        //         true => node!(
        //             u;
        //             attr!("shape", "doublecircle")
        //         ),
        //         false => node!(u),
        //     };
        //     g.add_stmt(Stmt::Node(node));
        //     for e in n.actions.iter() {
        //         // todo!();
        //         let edge = match e.next_position.as_ref() {
        //             Some(NextPositionData { next_position, .. }) => {
        //                 todo!();
        //                 // match self.nodes[next_position.get()].is_exhausted() {
        //                 //     true => edge!(
        //                 //         node_id!(u) => node_id!(next_position.get())
        //                 //     ),
        //                 //     false => edge!(
        //                 //         node_id!(u) => node_id!(next_position.get());
        //                 //         attr!("dir", "forward")
        //                 //     ),
        //                 // }
        //             },
        //             None => continue,
        //             // (None, None) => continue,
        //             // (None, Some(_)) => continue,
        //             // (Some(v), None) => edge!(
        //             //     node_id!(u) => node_id!(v)
        //             // ),
        //             // (Some(v), Some(_)) => edge!(
        //             //     node_id!(u) => node_id!(v);
        //             //     attr!("dir", "forward")
        //             // ),
        //         };
        //         let e = Stmt::Edge(edge);
        //         g.add_stmt(e);
        //     }
        // }
        let graph_svg = exec(g, &mut PrinterContext::default(), vec![Format::Png.into()]).unwrap();
        graph_svg
    }
}
