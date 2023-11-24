use std::collections::{BTreeMap, BTreeSet};

use az_discrete_opt::log::ShortForm;

use super::RootedOrderedTree;

impl<const N: usize> ShortForm for RootedOrderedTree<N> {
    fn short_form(&self) -> String {
        // if `i` is a parent node, concatenate with `i <- {children of i}`.
        let children = |p: usize| -> Option<String> {
            let mut children = self.parents().iter().enumerate().filter(|(_, &parent)| parent == p).map(|(i, _)| i);
            let first = children.next()?;
            let mut s = format!("{p} <- {{{first}");
            for child in children {
                s.push_str(&format!(", {child}"));
            }
            s.push_str("}, ");
            Some(s)
        };
        String::from_iter((0..N).filter_map(|i| children(i)))
    }
}

#[cfg(test)]
mod tests {
    use az_discrete_opt::log::ShortForm;

    use crate::rooted_tree::RootedOrderedTree;

    #[test]
    fn print_short_form_of_star_on_5_vertices() {
        let star = RootedOrderedTree::<5>::try_from([0, 0, 0, 0, 0]).unwrap();
        let short_form = star.short_form();
        println!("{short_form}");
    }

    #[test]
    fn print_short_form_of_this_one_tree_on_five_vertices() {
        let tree = RootedOrderedTree::<5>::try_from([0, 0, 1, 0, 1]).unwrap();
        let short_form = tree.short_form();
        println!("{short_form}");
    }
}