// import from tensor.rs Tensor
// disable unused imports
#![allow(unused_imports)]
mod autodiff;
mod operations;
pub mod tensor;
mod utils;
use tensor::Tensor;

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

#[cfg(test)]
mod tests {
    use super::*;

    mod ops {
        use super::*;

        #[test]
        fn test_addition() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
            let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
            let result = &a + &b;
            let expected = Tensor::new(vec![5.0, 7.0, 9.0], vec![3]);
            assert_eq!(result, expected);
        }

        #[test]
        fn test_transpose() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
            let result = a.transpose();
            let expected = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]);
            assert_eq!(result, expected);
        }
    }

    mod reductions {
        use super::*;

        #[test]
        fn test_sum() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
            let result = a.sum();
            assert_eq!(result, Tensor::new(vec![6.0], vec![]));
        }

        #[test]
        fn test_mean() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
            let result = a.mean();
            assert_eq!(result, Tensor::new(vec![2.0], vec![]));
        }
    }

    mod back {
        use super::*;

        #[test]
        fn test_add_back() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
            let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
            let result = &a + &b;
            let sum = result.sum();
            sum.backward();
            let grad = a.grad.borrow();
            if let Some(g) = grad.as_ref() {
                assert_eq!(g, &vec![1.0, 1.0, 1.0]);
            } else {
                panic!("Gradient is None");
            }

            let grad = b.grad.borrow();
            if let Some(g) = grad.as_ref() {
                assert_eq!(g, &vec![1.0, 1.0, 1.0]);
            } else {
                panic!("Gradient is None");
            }
        }
    }
}
