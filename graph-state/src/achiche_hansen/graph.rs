use core::marker::PhantomData;

#[derive(Clone)]
pub(crate) struct BoolEdges<const E: usize, Connectivity = ()> {
    pub(crate) edges: [bool; E],
    connected: PhantomData<Connectivity>,
}

#[derive(Clone)]
pub struct Neighborhoods<const N: usize, Connectivity = ()> {
    neighborhoods: [u32; N],
    connected: PhantomData<Connectivity>,
}

#[derive(Clone)]
pub enum Connected {}


#[derive(Clone)]
pub enum Disconnected {}

impl<const E: usize, C> BoolEdges<E, C> {
    fn edges(&self) -> &[bool; E] {
        &self.edges
    }

    fn forget_connectivity(self) -> BoolEdges<E, ()> {
        BoolEdges {
            edges: self.edges,
            connected: PhantomData,
        }
    }
}

impl<const N: usize> Neighborhoods<N, ()> {
    pub fn new(neighborhoods: [u32; N]) -> Self {
        Self {
            neighborhoods,
            connected: PhantomData,
        }
    }

    pub fn block_tree(&self) -> Option<BlockForest<N, Connected>> {
        todo!()
    }

    pub unsafe fn assert_connected(self) -> Neighborhoods<N, Connected> {
        Neighborhoods {
            neighborhoods: self.neighborhoods,
            connected: PhantomData,
        }
    }
}

impl<const N: usize, C> Neighborhoods<N, C> {
    pub fn distance_matrix(&self, blocks: &BlockForest<N, C>) -> DistanceMatrix<N, C> {
        todo!()
    }

    pub fn forget_connectivity(self) -> Neighborhoods<N, ()> {
        Neighborhoods {
            neighborhoods: self.neighborhoods,
            connected: PhantomData,
        }
    }

    pub fn cut_edges(&self, blocks: &BlockForest<N, C>) -> Vec<(usize, usize)> {
        todo!()
    }
}

impl<const E: usize> BoolEdges<E, ()> {
    pub fn new(edges: [bool; E]) -> Self {
        Self {
            edges,
            connected: PhantomData,
        }
    }
    
    pub unsafe fn assert_connected(self) -> BoolEdges<E, Connected> {
        BoolEdges {
            edges: self.edges,
            connected: PhantomData,
        }
    }
}

impl<const E: usize> BoolEdges<E, ()> {
    pub fn complement(self) -> BoolEdges<E> {
        BoolEdges {
            edges: self.edges.map(|b| !b),
            connected: PhantomData,
        }
    }
}

impl<const E: usize> BoolEdges<E, Connected> {
    pub fn complement(self) -> BoolEdges<E> {
        BoolEdges {
            edges: self.forget_connectivity().edges,
            connected: PhantomData,
        }
    }
}

impl<const E: usize> BoolEdges<E, Disconnected> {
    pub fn complement(self) -> BoolEdges<E, Connected> {
        BoolEdges {
            edges: self.edges,
            connected: PhantomData,
        }
    }
}

#[derive(Clone)]
pub struct BlockForest<const N: usize, Connectivity = ()> {
    connected: PhantomData<Connectivity>,
}

#[derive(Clone)]
pub struct DistanceMatrix<const N: usize, C> {
    connected: PhantomData<C>,
}
