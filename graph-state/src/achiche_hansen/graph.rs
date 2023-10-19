use core::marker::PhantomData;

#[derive(Clone)]
pub(crate) struct BoolEdges<const E: usize, Connectivity = ()> {
    pub(crate) edges: [bool; E],
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

impl<const E: usize> BoolEdges<E, ()> {
    pub fn generate<R: rand::Rng>(rng: &mut R, p: f64) -> Self {
        Self {
            edges: core::array::from_fn(|_| rng.gen_bool(p)),
            connected: PhantomData,
        }
    }

    pub fn to_connected_graph_with_blocks<const N: usize>(self) -> Option<(BoolEdges<E, Connected>, BlockForest<N, Connected>)> {
        todo!()
    }
}

impl<const E: usize> BoolEdges<E, ()> {
    pub fn complement(self) -> BoolEdges<E> {
        let Self {
            edges,
            connected: _,
        } = self;
        BoolEdges {
            edges: edges.map(|b| !b),
            connected: Default::default(),
        }
    }
}

impl<const E: usize> BoolEdges<E, Connected> {
    pub fn complement(self) -> BoolEdges<E> {
        let edges = self.forget_connectivity().edges;
        BoolEdges {
            edges,
            connected: Default::default(),
        }
    }
}

impl<const E: usize> BoolEdges<E, Disconnected> {
    pub fn complement(self) -> BoolEdges<E, Connected> {
        let edges = self.forget_connectivity().edges;
        BoolEdges {
            edges,
            connected: Default::default(),
        }
    }
}

#[derive(Clone)]
pub struct BlockForest<const N: usize, Connectivity = ()> {
    connected: PhantomData<Connectivity>,
}

#[derive(Clone)]
pub struct DistanceMatrix<const N: usize> {

}
