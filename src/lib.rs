// import from tensor.rs Tensor
// disable unused imports
#![allow(unused_imports)]
mod tensor;
use std::cell::RefCell;
use std::panic;
use tensor::Tensor;
use tensor::TensorRef;

#[cfg(test)]
mod tests {
    use super::*;

    // #[test]
    fn tensor_ops() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);

        assert_eq!(&a + &b, Tensor::new(vec![5.0, 7.0, 9.0], vec![3]));
        assert_eq!(&a - &b, Tensor::new(vec![-3.0, -3.0, -3.0], vec![3]));
        assert_eq!(&a * &b, Tensor::new(vec![4.0, 10.0, 18.0], vec![3]));
        assert_eq!(&a / &b, Tensor::new(vec![0.25, 0.4, 0.5], vec![3]));
    }

    // #[test]
    fn matmul() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
        let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0], vec![2, 2]);
        let result = a.matmul(&b);
        assert_eq!(result.shape, vec![2, 2]);
        assert_eq!(result.data, vec![19.0, 22.0, 43.0, 50.0]);
    }

    // #[test]
    fn sum() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let sum = a.sum();
        assert_eq!(sum.data, vec![6.0]);
        assert_eq!(sum.shape, vec![]);
    }

    // #[test]
    fn sum_back() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let sum = a.sum();
        sum.backward();
        let correct_grad = RefCell::new(Some(vec![1.0, 1.0, 1.0]));
        assert_eq!(a.grad, correct_grad,);
    }

    // #[test]
    fn add_back() {
        // test addition backprop
        // requires grad by default
        let a: TensorRef = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        assert_eq!(a.requires_grad, true);
        let b: TensorRef = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        assert_eq!(b.requires_grad, true);

        let c = &a + &b;

        let sum = c.sum();

        sum.backward();

        assert_eq!(*a.grad.borrow(), Some(vec![1.0, 1.0, 1.0]));
    }

    #[test]
    fn mul_back() {
        let a: TensorRef = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b: TensorRef = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);

        let c = &a * &b;

        let sum = c.sum();

        sum.backward();

        // we expect the a.grad tensor to be full of 1.0s
        assert_eq!(*a.grad.borrow(), Some(vec![4.0, 5.0, 6.0]));
        assert_eq!(*b.grad.borrow(), Some(vec![1.0, 2.0, 3.0]));
    }
}
