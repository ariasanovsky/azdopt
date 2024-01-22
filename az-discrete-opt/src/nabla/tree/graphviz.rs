use dot_generator::*;
use dot_structures::*;
use graphviz_rust::{cmd::Format, exec, printer::PrinterContext};
use petgraph::visit::IntoNodeReferences;

use super::SearchTree;

impl<P> SearchTree<P> {
    pub fn graphviz(&self) -> Vec<u8> {
        let mut g = graph!(id!("search_tree"));
        for (u, n) in self.tree.node_references() {
            let u_id = u.index();
            let node = match n.n_t.try_active().is_some() {
                false => node!(
                    u_id;
                    attr!("shape", "doublecircle")
                ),
                true => node!(u_id),
            };
            g.add_stmt(Stmt::Node(node));
        }
        for u in self.tree.node_indices() {
            let u_id = u.index();
            for v in self
                .tree
                .neighbors_directed(u, petgraph::Direction::Outgoing)
            {
                let v_id = v.index();
                let edge = match self.tree[v].n_t.try_active().is_some() {
                    false => edge!(
                        node_id!(u_id) => node_id!(v_id)
                    ),
                    true => edge!(
                        node_id!(u_id) => node_id!(v_id);
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
