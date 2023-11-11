use az_discrete_opt::log::ShortForm;

use super::Tree;

impl<const N: usize> ShortForm for Tree<N> {
    fn short_form(&self) -> String {
        let mut edges = self.edges().collect::<Vec<_>>();
        edges.sort_by_key(|e| e.max());
        format!("{edges:?}")
    }
}

impl<const N: usize> Tree<N> {
    /// g6 standard: http://users.cecs.anu.edu.au/~bdm/data/formats.html
    /// http://users.cecs.anu.edu.au/~bdm/data/formats.txt
    pub fn to_sparse6(&self) -> Vec<u8> {
        let mut sparse6: Vec<u8> = Vec::new();

        // Add the number of vertices to the sparse6 string
        if N <= 62 {
            sparse6.push(N as u8 + 63);
        } else {
            unimplemented!("graphs on more than 62 vertices")
            // sparse6.push((126 as u8) as char);
            // sparse6.push(((N >> 12) + 63) as char);
            // sparse6.push((((n >> 6) & 63) + 63) as char);
            // sparse6.push(((n & 63) + 63) as char);
        }
        // first, let `k` be the number of bits needed to represent `n - 1`
        // the edges are encoded in a sequences that looks like (b_i x_i)_{i=0..l}
        // each `b` is a single bit, each `x` is `k` bits
        // `v` starts at `0` and we read the next (b x) from the front
        // `b` indicates if `v` increments by `1`
        // we then compare `v` to `x` and either replace `v` with `x` or treat `{v, x}` as an edge
        let k = (N - 1).ilog2();
        for edge in self.edges() {
            todo!()
        }
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn sparse6_documentation_example_is_correct() {
        let correct_s6 = b":Fa@x^";
        // edge set: 0-1 0-2 1-2 5-6
    }
}