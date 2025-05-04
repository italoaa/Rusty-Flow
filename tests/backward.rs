use rflow::tensor::Tensor;
// backward.rs

#[cfg(test)]
mod backward {
    use super::*;

    // Tests for backward pass of tensor operations
    mod backward_ops {
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
            assert_eq!(agrad, &vec![11.0, 15.0, 11.0, 15.0]);
            let bgrad = b.grad.borrow();
            let bgrad = bgrad.as_ref().unwrap();
            assert_eq!(bgrad, &vec![4.0, 4.0, 6.0, 6.0]);
        }

        #[test]
        fn test_relu_back() {
            let a = Tensor::new(vec![-1.0, 2.0, -3.0, 4.0], vec![4]);
            let result = a.relu();
            let sum = result.sum();
            sum.backward();
            let agrad = a.grad.borrow();
            let agrad = agrad.as_ref().unwrap();
            assert_eq!(agrad, &vec![0.0, 1.0, 0.0, 1.0]);
        }

        #[test]
        fn test_mse_back() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
            // B is the target normally does not require grad
            let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
            let result = a.mse(&b);
            let sum = result.sum();
            sum.backward();
            let agrad = a.grad.borrow();
            let agrad = agrad.as_ref().unwrap();
            assert_eq!(agrad, &vec![-6.0, -6.0, -6.0]);
        }

        #[test]
        fn test_ce_back() {
            let a = Tensor::new(vec![0.8, 0.1, 0.1], vec![3]);
            // B is the target normally does not require grad
            let b = Tensor::new(vec![0.0, 1.0, 0.0], vec![3]);
            let result = a.cross_entropy(&b);
            let sum = result.sum();
            sum.backward();
            let agrad = a.grad.borrow();
            let agrad = agrad.as_ref().unwrap();
            assert_eq!(agrad, &vec![-0.0, -10.0, -0.0]);
        }

        #[test]
        fn test_softmax_back() {
            let expected = Tensor::new(
                vec![
                    0.0819, -0.0220, -0.0599, -0.0220, 0.1848, -0.1628, -0.0599, -0.1628, 0.2227,
                ],
                vec![3, 3],
            );
            for i in 0..3 {
                let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
                let soft = a.softmax();
                let vecmask = if i == 0 {
                    vec![1.0, 0.0, 0.0]
                } else if i == 1 {
                    vec![0.0, 1.0, 0.0]
                } else {
                    vec![0.0, 0.0, 1.0]
                };
                let mask = Tensor::new(vecmask, vec![3]);
                let masked = &soft * &mask;
                let loss = masked.sum();
                loss.backward();
                let agrad = a.grad.borrow();
                let agrad = agrad.as_ref().unwrap();
                let expected_grad =
                    Tensor::new(expected.data[i * 3..(i + 1) * 3].to_vec(), vec![3]);
                let mse = a.mse(&expected_grad);
                for j in 0..agrad.len() {
                    assert!(
                        mse.data[j] > 1e-6,
                        "Expected {} to be close to {}",
                        agrad[j],
                        expected_grad.data[j]
                    );
                }
            }
        }
    }
}
