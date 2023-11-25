#![feature(slice_flatten)]

use az_discrete_opt::space::StateActionSpace;

#[derive(Debug, Clone)]
pub struct GreedyGrid<const N: usize> {
    tiles: [[Tile; N]; N],
}

impl<const N: usize> Default for GreedyGrid<N> {
    fn default() -> Self {
        Self {
            tiles: [[Tile::Free; N]; N],
        }
    }
}

impl<const N: usize> GreedyGrid<N> {
    const OCTANT: FirstOctantSlopesConstrained<N> = FirstOctantSlopesConstrained::new();

    pub(crate) fn tile_iter(&self) -> impl Iterator<Item = &Tile> + '_ {
        self.tiles.iter().flat_map(|row| row.iter())
    }

    pub fn cardinality(&self) -> usize {
        self.tile_iter().filter(|tile| {
            if let Tile::Selected = tile {
                true
            } else {
                false
            }
        }).count()
    }
}

#[derive(Debug, Clone, Copy, Default)]
enum Tile {
    #[default]
    Free,
    Selected,
    Prohibited,
}

pub enum ModifyTile {
    Select { x: usize, y: usize },
}

pub struct ThreeInALineMVP<const N: usize>;

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
                tiles[*x][*y] = Tile::Selected;
                let non_degenerate_slopes = Self::State::OCTANT.elements().iter().flat_map(|(a, b)| {
                    [
                        CanonicalSlope::new(*a, *b as _),
                        CanonicalSlope::new(*a, -(*b as isize)),
                        CanonicalSlope::new(*b, *a as _),
                        CanonicalSlope::new(*b, -(*a as isize)),
                    ].into_iter()
                }).filter(|slope| {
                    // get the other points on the line
                    // check if one of them is selected
                    slope.other_points_within_grid((*x, *y), N).any(|(x, y)| {
                        if let Tile::Selected = tiles[x][y] {
                            true
                        } else {
                            false
                        }
                    })
                }).collect::<Vec<_>>();
                // mark all points on the lines as prohibited
                for slope in non_degenerate_slopes {
                    for (x, y) in slope.other_points_within_grid((*x, *y), N) {
                        tiles[x][y] = Tile::Prohibited;
                    }
                }
                let prohibit_along_horizontal = (0..N).any(|y| {
                    if let Tile::Selected = tiles[*x][y] {
                        true
                    } else {
                        false
                    }
                });
                if prohibit_along_horizontal {
                    for y in 0..N {
                        tiles[*x][y] = Tile::Prohibited;
                    }
                }
                let prohibit_along_vertical = (0..N).any(|x| {
                    if let Tile::Selected = tiles[x][*y] {
                        true
                    } else {
                        false
                    }
                });
                if prohibit_along_vertical {
                    for x in 0..N {
                        tiles[x][*y] = Tile::Prohibited;
                    }
                }
                let diagonal = CanonicalSlope::new(1, 1);
                let prohibit_along_diagonal = diagonal.other_points_within_grid((*x, *y), N).any(|(x, y)| {
                    if let Tile::Selected = tiles[x][y] {
                        true
                    } else {
                        false
                    }
                });
                if prohibit_along_diagonal {
                    for (x, y) in diagonal.other_points_within_grid((*x, *y), N) {
                        tiles[x][y] = Tile::Prohibited;
                    }
                }
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

// const fn first_positive_octant_excluding_1_1<const O: usize>(n: usize) -> [(usize, usize); O] {
//     let mut octant = [(0, 0); O];
//     let mut i = 0;
//     let mut x = 2;
//     while x < n {
//         let mut y = 1;
//         while y < x {
//             if are_coprime(x, y) {
//                 octant[i] = (x, y);
//                 i += 1;
//             }
//             y += 1;
//         }
//         x += 1;
//     }
//     assert!(
//         i >= O,
//         "The generic number `O` of generators in the first octant is too large."
//     );
//     assert!(
//         i <= O,
//         "The generic number `O` of generators in the first octant is too small."
//     );
//     octant
// }

const fn are_coprime(mut a: usize, mut b: usize) -> bool {
    while b != 0 {
        let r = a % b;
        a = b;
        b = r;
    }
    a == 1
}

/// Calculate the set of generators of lines in the lattice.
/// We reduce this set up to symmetry and are left with:
/// `O_N = {(x, y) | 1 <= y < x < N, gcd(x, y) = 1}`.
struct FirstOctantSlopesConstrained<const N: usize> {
    slopes: [[(usize, usize); N]; N],
    len: usize,
}

impl<const N: usize> FirstOctantSlopesConstrained<N> {
    const fn new() -> Self {
        let mut octant = [[(0, 0); N]; N];
        let mut i = 0;
        let mut x = 2;
        while x <= (N  - 1) / 2 {
            let mut y = 1;
            while y < x {
                if are_coprime(x, y) {
                    octant[i / N][i % N] = (x, y);
                    i += 1;
                }
                y += 1;
            }
            x += 1;
        }
        Self {
            slopes: octant,
            len: i,
        }
    }

    fn elements(&self) -> &[(usize, usize)] {
        let flat = self.slopes.flatten();
        &flat[..self.len]
    }
}

struct CanonicalSlope {
    a: usize,
    b: isize,
}

impl CanonicalSlope {
    fn new(a: usize, b: isize) -> Self {
        Self { a, b }
    }

    fn other_points_within_grid(
        &self,
        x_0: (usize, usize),
        n: usize,
    ) -> impl Iterator<Item = (usize, usize)>  + '_ {
        // calculate the maximum and minimum offsets which correspond to points inside `{0, ..., n-1}^2`
        let (a, b) = (self.a, self.b);
        let (x, y) = x_0;
        let mut max_u = (n - 1 - x) / a;
        let mut min_u = - ((x / a) as isize);
        match b.cmp(&0) {
            std::cmp::Ordering::Less => {
                max_u = max_u.min(y / (-b) as usize);
                min_u = min_u.max(
                    -((
                        (n - 1 - y) / (-b) as usize
                    ) as isize)
                );
            },
            std::cmp::Ordering::Equal => unimplemented!(),
            std::cmp::Ordering::Greater => {
                max_u = max_u.min((n - 1 - y) / b as usize);
                min_u = min_u.max(-((y / b as usize) as isize));
            },
        }
        let points_ahead = (1..=max_u).map(move |u| (x + u * a, (y as isize + u as isize * b) as usize));
        let points_behind = (min_u..0).map(move |u| (
            (x as isize + u * (a as isize)) as _,
            (y as isize + u * b) as _,
        ));
        points_ahead.chain(points_behind)
    }
        
}

#[cfg(test)]
mod tests {
    use crate::CanonicalSlope;

    use super::FirstOctantSlopesConstrained;
    #[test]
    fn first_ocant_n_equals_5_is_correct() {
        let correct_first_octant = [
            (2, 1),
            // (3, 1),
            // (3, 2),
            // (4, 1),
            // (4, 3),
        ];
        let first_octant = FirstOctantSlopesConstrained::<5>::new();
        assert_eq!(first_octant.elements(), correct_first_octant);
    }

    #[test]
    fn first_ocant_n_equals_8_is_correct() {
        let correct_first_octant = [
            (2, 1),
            (3, 1),
            (3, 2),
            // (4, 1),
            // (4, 3),
            // (5, 1),
            // (5, 2),
            // (5, 3),
            // (5, 4),
            // (6, 1),
            // (6, 5),
            // (7, 1),
            // (7, 2),
            // (7, 3),
            // (7, 4),
            // (7, 5),
            // (7, 6),
        ];
        let first_octant = FirstOctantSlopesConstrained::<8>::new();
        assert_eq!(first_octant.elements(), correct_first_octant);
    }

    #[test]
    fn the_line_with_slope_1_2_through_2_2_is_correct_in_grid_10() {
        let correct_line = [
            (1, 0),
            (3, 4),
            (4, 6),
            (5, 8),
        ];
        let slope = CanonicalSlope::new(1, 2);
        let mut line = slope.other_points_within_grid((2, 2), 10).collect::<Vec<_>>();
        line.sort();
        assert_eq!(line, correct_line);
    }

    #[test]
    fn the_line_with_slope_2_negative_1_through_3_3_is_correct_in_grid_10() {
        let correct_line = [
            (1, 4),
            (5, 2),
            (7, 1),
            (9, 0),
        ];
        let slope = CanonicalSlope::new(2, -1);
        let mut line = slope.other_points_within_grid((3, 3), 10).collect::<Vec<_>>();
        line.sort();
        assert_eq!(line, correct_line);
    }
}
