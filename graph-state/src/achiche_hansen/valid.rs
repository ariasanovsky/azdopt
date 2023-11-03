// use crate::simple_graph::neighborhoods::Neighborhoods;

pub(crate) trait GenericsAreValid {
    const VALID: ();
}

// impl<const N: usize> GenericsAreValid for Neighborhoods<N> {
//     const VALID: () = {
//         assert!(N <= 32, "N must be <= 32");
//         ()
//     };
// }
