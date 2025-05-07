// import from tensor.rs Tensor
// disable unused imports
#![allow(unused_imports)]
mod autodiff;
pub mod broadcast;
mod operations;
pub mod optimizers;
pub mod tensor;
mod utils;
use broadcast::{broadcast_shape, BroadcastIterator};

use std::sync::atomic::{AtomicBool, Ordering};

// DEBUGGING ---
static DEBUG: AtomicBool = AtomicBool::new(false);

pub fn set_debug(value: bool) {
    DEBUG.store(value, Ordering::Relaxed);
}

fn is_debug() -> bool {
    DEBUG.load(Ordering::Relaxed)
}
// DEBUGGING ---

// make a main function to test the library
// for now just print hello world
