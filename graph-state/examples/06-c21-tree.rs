use std::io::BufWriter;

use dfdx::nn::{modules::ReLU, builders::Linear};
use tensorboard_writer::TensorboardWriter;

const STATE: usize = 1;
const ACTION: usize = 1;

const HIDDEN_1: usize = 512;
const HIDDEN_2: usize = 1024;
const HIDDEN_3: usize = 512;

type ModelH = (
    (Linear<STATE, HIDDEN_1>, ReLU),
    (Linear<HIDDEN_1, HIDDEN_2>, ReLU),
    (Linear<HIDDEN_2, HIDDEN_3>, ReLU),
    // (Linear<HIDDEN_3, ACTION>, dfdx::nn::modules::Sigmoid),
    (Linear<HIDDEN_3, ACTION>, ReLU),
);

const BATCH: usize = 512;

type W = TensorboardWriter<BufWriter<std::fs::File>>;

fn main() -> eyre::Result<()> {
    todo!()
}
