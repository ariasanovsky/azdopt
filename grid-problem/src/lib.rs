use az_discrete_opt::space::StateActionSpace;

pub struct GreedyGrid<const N: usize> {
    tiles: [[Tile; N]; N],
}

impl<const N: usize> GreedyGrid<N> {
    pub(crate) fn tile_iter(&self) -> impl Iterator<Item = &Tile> + '_ {
        self.tiles.iter().flat_map(|row| row.iter())
    }
    
}

enum Tile {
    Free,
    Selected,
    Prohibited,
}

enum ModifyTile {
    Select { x: usize, y: usize },
}

struct ThreeInALineMVP<const N: usize>;

impl<const N: usize> StateActionSpace for ThreeInALineMVP<N> {
    type State = GreedyGrid<N>;

    type Action = ModifyTile;

    /// The dimension of the state vector.
    /// We consider the direct sum of two boolean vectors.
    /// The first vector indicates whether a tile is selected.
    /// The second vector indicates whether a tile is prohibited.
    const DIM: usize = 2 * N * N;

    fn index(action: &Self::Action) -> usize {
        match action {
            ModifyTile::Select { x, y } => x * N + y,
        }
    }

    fn from_index(index: usize) -> Self::Action {
        let x = index / N;
        let y = index % N;
        ModifyTile::Select { x, y }
    }

    fn act(state: &mut Self::State, action: &Self::Action) {
        let tiles = &mut state.tiles;
        match action {
            ModifyTile::Select { x, y } => {
                state.tiles[*x][*y] = Tile::Selected;
                todo!("mark some `Free` tiles as `Prohibited`");
            }
        }
    }

    fn action_indices(state: &Self::State) -> impl Iterator<Item = usize> {
        state.tiles.iter().enumerate().flat_map(|(x, row)| {
            row.iter().enumerate().filter_map(move |(y, tile)| {
                if let Tile::Free = tile {
                    let action = ModifyTile::Select { x, y };
                    let index = Self::index(&action);
                    Some(index)
                } else {
                    None
                }
            })
        })
    }

    /// Split the vector into a `Self::DIM / 2` dimensional vector of pairs.
    /// For each pair (a, b), the first component is whether the tile is selected.
    /// The second component is whether the tile is prohibited.
    fn write_vec(state: &Self::State, vec: &mut [f32]) {
        debug_assert_eq!(vec.len(), Self::DIM);
        let tiles = state.tile_iter();
        let chunks = vec.chunks_exact_mut(2);
        for (tile, chunk) in tiles.zip(chunks) {
            match tile {
                Tile::Free => {
                    chunk[0] = 0.0;
                    chunk[1] = 0.0;
                }
                Tile::Selected => {
                    chunk[0] = 1.0;
                    chunk[1] = 0.0;
                }
                Tile::Prohibited => {
                    chunk[0] = 0.0;
                    chunk[1] = 1.0;
                }
            }
        }
    }

    fn follow(
        state: &mut Self::State,
        actions: impl Iterator<Item = Self::Action>,
    ) {
        for action in actions {
            Self::act(state, &action);
        }
    }

    fn is_terminal(state: &Self::State) -> bool {
        Self::action_indices(state).next().is_none()
    }

    fn has_action(state: &Self::State, action: &Self::Action) -> bool {
        let ModifyTile::Select { x, y } = action;
        let tiles = &state.tiles;
        if let Tile::Free = tiles[*x][*y] {
            true
        } else {
            false
        }
    }
}

/// Calculate the set of generators of lines in the lattice.
/// We reduce this set up to symmetry and are left with:
/// `O_N = {(x, y) | 1 <= y < x < N, gcd(x, y) = 1}`.
const fn first_positive_octant_excluding_1_1<const O: usize>(n: usize) -> [(usize, usize); O] {
    let mut octant = [(0, 0); O];
    let mut i = 0;
    let mut x = 2;
    while x < n {
        let mut y = 1;
        while y < x {
            if are_coprime(x, y) {
                octant[i] = (x, y);
                i += 1;
            }
            y += 1;
        }
        x += 1;
    }
    assert!(
        i >= O,
        "The generic number `O` of generators in the first octant is too large."
    );
    assert!(
        i <= O,
        "The generic number `O` of generators in the first octant is too small."
    );
    octant
}

const fn are_coprime(mut a: usize, mut b: usize) -> bool {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a == 1
}

#[cfg(test)]
mod tests {
    use super::first_positive_octant_excluding_1_1;
    #[test]
    fn first_ocant_n_equals_5_is_correct() {
        let correct_first_octant = [
            (2, 1),
            (3, 1),
            (3, 2),
            (4, 1),
            (4, 3),
        ];
        let first_octant = first_positive_octant_excluding_1_1::<5>(5);
        assert_eq!(first_octant, correct_first_octant);
    }
}

// struct Foo<const N: usize>;

// const fn double(n: usize) -> usize {
//     n * 2
// }