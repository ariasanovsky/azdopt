use dot_generator::*;
use dot_structures::*;
use graphviz_rust::{printer::PrinterContext, cmd::Format, exec};

use super::SearchTree;

impl<P> SearchTree<P> {
    pub fn graphviz(&self) -> Vec<u8> {
        let mut g = graph!(
            id!("search_tree")
        );

        for (u, n) in self.nodes.iter().enumerate() {
            let node = match n.is_exhausted() {
                true => node!(
                    u;
                    attr!("shape", "doublecircle")
                ),
                false => node!(u)
            };
            g.add_stmt(Stmt::Node(node));
            for e in n.actions.iter() {
                let edge = match (e.next_position(), e.g_sa()) {
                    (None, None) => continue,
                    (None, Some(_)) => continue,
                    (Some(v), None) => edge!(
                        node_id!(u) => node_id!(v)
                    ),
                    (Some(v), Some(_)) => edge!(
                        node_id!(u) => node_id!(v);
                        attr!("dir", "forward")
                    ),
                };
                let e = Stmt::Edge(edge);
                g.add_stmt(e);
            }
        }


        // let edges = [
        //     (0, 1),
        //     (1, 2),
        //     (2, 3)
        // ];
        // for (a, b) in edges.iter() {
        //     let e = edge!(
        //         node_id!(esc a) => node_id!(esc b);
        //         attr!("color", "red"),
        //         attr!("dir", "forward"),
        //         attr!("style", "dashed")
        //     );
        //     let e = Stmt::Edge(e);
        //     g.add_stmt(e);
        // }
        let graph_svg = exec(
            g,
            &mut PrinterContext::default(),
            vec![Format::Svg.into()],
        )
        .unwrap();
    graph_svg
    }
}