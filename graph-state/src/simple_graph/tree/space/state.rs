use crate::simple_graph::tree::PrueferCode;

impl<const N: usize> PrueferCode<N> {
    pub fn modify_entry(&mut self, index: usize, new_parent: usize) {
        self.code[index] = new_parent
    }
}
