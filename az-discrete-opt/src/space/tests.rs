use crate::{
    int_min_tree::{
        state_data::{action_data::INTUnvisitedActionData, INTStateData, UpperEstimateData},
        INTMinTree,
    },
    path::{sequence::ActionSequence, ActionPath},
};

use super::StateActionSpace;

struct MinimizeNSquared<const N: i32>;

#[derive(Debug, Clone)]
struct Interval<const N: i32>(i32);

#[derive(Debug)]
enum PlusOrMinusOne {
    Minus,
    Plus,
}

impl<const N: i32> StateActionSpace for MinimizeNSquared<N> {
    type State = Interval<N>;

    type Action = PlusOrMinusOne;

    const DIM: usize = 1;

    fn index(action: &Self::Action) -> usize {
        match action {
            PlusOrMinusOne::Minus => 0,
            PlusOrMinusOne::Plus => 1,
        }
    }

    fn from_index(index: usize) -> Self::Action {
        match index {
            0 => PlusOrMinusOne::Minus,
            1 => PlusOrMinusOne::Plus,
            _ => panic!("invalid index"),
        }
    }

    fn act(state: &mut Self::State, action: &Self::Action) {
        match (action, state) {
            (PlusOrMinusOne::Minus, Interval(n)) if *n < 5 => *n -= 1,
            (PlusOrMinusOne::Plus, Interval(n)) if *n > -5 => *n += 1,
            (a, s) => panic!("invalid action {a:?} for state {s:?}"),
        }
    }

    fn actions(state: &Self::State) -> impl Iterator<Item = usize> {
        match state.0 {
            n if n == N => vec![0],
            n if n == -N => vec![1],
            _ => vec![0, 1],
        }
        .into_iter()
    }

    fn write_vec(_state: &Self::State, _vec: &mut [f32]) {
        todo!()
    }
}

#[test]
fn transitions_for_minimizing_n_square_are_correct() {
    type Space = MinimizeNSquared<5>;
    type P = ActionSequence;
    type S = Interval<5>;
    // type A = PlusOrMinusOne;
    type Tree = INTMinTree<P>;

    let state_to_prediction_index = |s: &S| match s {
        Interval(n) => (*n + 5) as usize,
    };

    let prediction = |s: &S| {
        let predictions: [[f32; 2]; 11] = [
            [0.05, 0.95],
            [0.15, 0.85],
            [0.25, 0.75],
            [0.35, 0.65],
            [0.45, 0.55],
            [0.50, 0.50],
            [0.55, 0.45],
            [0.65, 0.35],
            [0.75, 0.25],
            [0.85, 0.15],
            [0.95, 0.05],
        ];
        predictions[state_to_prediction_index(s)]
    };

    let cost = |s: &S| {
        let sq = s.0 * s.0;
        sq as f32
    };

    let upper_estimate = |estimate: UpperEstimateData| {
        let UpperEstimateData {
            n_s: _,
            n_sa,
            g_sa_sum: _,
            p_sa,
            depth: _,
        } = estimate;
        debug_assert_ne!(n_sa, 0);
        p_sa
    };

    let s_0: S = Interval(-3);

    let mut tree = Tree::new::<Space>(&prediction(&s_0), cost(&s_0), &s_0);
    let correct_root_data = INTStateData {
        n_s: 0,
        c_star: 9.0,
        visited_actions: vec![],
        unvisited_actions: vec![
            INTUnvisitedActionData { a: 0, p_sa: 0.25 },
            INTUnvisitedActionData { a: 1, p_sa: 0.75 },
        ],
    };
    assert_eq!(&tree.root_data, &correct_root_data);
    assert!(tree.data.is_empty());

    let mut s_t = s_0.clone();
    let mut p_t = P::new();
    // let mut n_0 = MutRefNode::new(&mut s_t, &mut p_t);
    let mut transitions = tree.simulate_once::<Space>(&mut s_t, &mut p_t, &upper_estimate);
    // dbg!(&transitions, &s_t, &p_t);
    // transitions.update_existing_nodes(c_t, s_t, p_t, probs_t, g_star_theta_s_t)
    todo!("assert eq")
}

// let upper_estimate = |estimate: UpperEstimateData| {
//     let UpperEstimateData { n_s, n_sa, g_sa_sum, p_sa, depth } = estimate;
//     debug_assert_ne!(n_s, 0);
//     debug_assert_ne!(n_sa, 0);
//     let n_s = n_s as f32;
//     let n_sa = n_sa as f32;
//     let c_puct = 0.0;
//     let g_sa = g_sa_sum / n_sa;
//     let u_sa = g_sa + c_puct * p_sa * (n_s.sqrt() / n_sa);
//     // println!(
//     //     "{u_sa} = {g_sa_sum} / {n_sa} + {c_puct} * {p_sa} * ({n_s}.sqrt() / {n_sa})",
//     // );
//     u_sa
// };
