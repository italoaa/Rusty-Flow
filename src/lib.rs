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
static DEBUG: AtomicBool = AtomicBool::new(true);

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
        fn test_subtraction() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
            let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
            let result = &a - &b;
            let expected = Tensor::new(vec![-3.0, -3.0, -3.0], vec![3]);
            assert_eq!(result, expected);
        }

        #[test]
        fn test_multiplication() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
            let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
            let result = &a * &b;
            let expected = Tensor::new(vec![4.0, 10.0, 18.0], vec![3]);
            assert_eq!(result, expected);
        }

        #[test]
        fn test_division() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
            let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
            let result = &a / &b;
            let expected = Tensor::new(vec![0.25, 0.4, 0.5], vec![3]);
            assert_eq!(result, expected);
        }

        #[test]
        fn test_transpose() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
            let result = a.transpose();
            let expected = Tensor::new(vec![1.0, 3.0, 2.0, 4.0], vec![2, 2]);
            assert_eq!(result, expected);
        }

        #[test]
        fn test_mm() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
            let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
            let result = a.mm(&b);
            let expected = Tensor::new(vec![19.0, 22.0, 43.0, 50.0], vec![2, 2]);
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
            let agrad = a.grad.borrow();
            let agrad = agrad.as_ref().unwrap();
            assert_eq!(agrad, &vec![1.0, 1.0, 1.0]);
            let bgrad = b.grad.borrow();
            let bgrad = bgrad.as_ref().unwrap();
            assert_eq!(bgrad, &vec![1.0, 1.0, 1.0]);
        }

        #[test]
        fn test_sub_back() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
            let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
            let result = &a - &b;
            let sum = result.sum();
            sum.backward();
            let agrad = a.grad.borrow();
            let agrad = agrad.as_ref().unwrap();
            assert_eq!(agrad, &vec![1.0, 1.0, 1.0]);
            let bgrad = b.grad.borrow();
            let bgrad = bgrad.as_ref().unwrap();
            assert_eq!(bgrad, &vec![-1.0, -1.0, -1.0]);
        }

        #[test]
        fn test_mul_back() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
            let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
            let result = &a * &b;
            let sum = result.sum();
            sum.backward();
            let agrad = a.grad.borrow();
            let agrad = agrad.as_ref().unwrap();
            assert_eq!(agrad, &vec![4.0, 5.0, 6.0]);
            let bgrad = b.grad.borrow();
            let bgrad = bgrad.as_ref().unwrap();
            assert_eq!(bgrad, &vec![1.0, 2.0, 3.0]);
        }

        #[test]
        fn test_div_back() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
            let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
            let result = &a / &b;
            let sum = result.sum();
            sum.backward();
            let agrad = a.grad.borrow();
            let agrad = agrad.as_ref().unwrap();
            assert_eq!(agrad, &vec![0.25, 0.2, 0.16666667]);
            let bgrad = b.grad.borrow();
            let bgrad = bgrad.as_ref().unwrap();
            assert_eq!(bgrad, &vec![-0.0625, -0.08, -0.08333333333333333]);
        }

        #[test]
        fn test_sum_back() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
            let result = a.sum();
            result.backward();
            let agrad = a.grad.borrow();
            let agrad = agrad.as_ref().unwrap();
            assert_eq!(agrad, &vec![1.0, 1.0, 1.0]);
        }

        #[test]
        fn test_mean_back() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
            let result = a.mean();
            result.backward();
            let agrad = a.grad.borrow();
            let agrad = agrad.as_ref().unwrap();
            assert_eq!(agrad, &vec![1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]);
        }

        #[test]
        fn test_mm_back() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
            let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
            let result = a.mm(&b);
            let sum = result.sum();
            sum.backward();
            let agrad = a.grad.borrow();
            let agrad = agrad.as_ref().unwrap();
            assert_eq!(agrad, &vec![5.0, 7.0, 6.0, 8.0]);
            let bgrad = b.grad.borrow();
            let bgrad = bgrad.as_ref().unwrap();
            assert_eq!(bgrad, &vec![1.0, 2.0, 3.0, 4.0]);
        }
    }
}
