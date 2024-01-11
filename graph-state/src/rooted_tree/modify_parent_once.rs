use super::RootedOrderedTree;

pub struct ROTWithParentPermissions<const N: usize> {
    tree: RootedOrderedTree<N>,
    permissions: [bool; N],
}
