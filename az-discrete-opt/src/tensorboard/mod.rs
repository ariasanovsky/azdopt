use std::path::{PathBuf, Path};

use tensorboard_writer::proto::tensorboard::Summary;

pub fn tf_path() -> PathBuf {
    std::env::var("OUT_DIR")
        .map_or_else(
            |_| {
                std::env::var("CARGO_MANIFEST_DIR")
                    .map(|manifest_dir| Path::new(&manifest_dir).join("target"))
            },
            |out_dir| Ok(PathBuf::from(out_dir)),
        )
        .unwrap_or_else(|_| "/home/target".into())
        .join("tensorboard")
}

pub trait Summarize {
    fn summary(&self) -> Summary;
}