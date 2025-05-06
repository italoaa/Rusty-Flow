use rflow::broadcast::{broadcast_shape, BroadcastIterator};
use rflow::tensor::{Tensor, TensorRef};

#[cfg(test)]
mod others {
    use super::*;

    #[test]
    fn test_softmaxv2() {
        let a = Tensor::new(
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
            vec![2, 4, 2],
        );

        let softmax = a.softmax(1);
        let expected = Tensor::new(
            vec![
                0.0021, 0.0021, 0.0158, 0.0158, 0.1171, 0.1171, 0.8650, 0.8650, 0.0021, 0.0021,
                0.0158, 0.0158, 0.1171, 0.1171, 0.8650, 0.8650,
            ],
            vec![2, 4, 2],
        );

        // Check if the softmax is correct
        assert!(
            softmax.approx_eq(&expected, 1e-4),
            "Expected: {:?}, Got: {:?}",
            expected,
            softmax
        );
    }

    #[test]
    fn test_iter_over_dim() {
        // 16 elements
        let a = Tensor::new(
            vec![
                0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0,
                15.0,
            ],
            vec![2, 4, 2],
        );

        // Iterate over the first dimension
        //
        let mut iter = a.iterate_over_dim(2).enumerate();
        // print each element
        let expected: Vec<f32> = vec![1., 5., 9., 13., 17., 21., 25., 29.];
        while let Some((i, group)) = iter.next() {
            // print the sum of the group
            let sum: f32 = group.iter().sum();
            // println!("Group {}: {:?} => Sum: {}", i, group, sum);
            assert_eq!(sum, expected[i]);
        }
    }

    /// The aim of these tests is to decompose trivial neural network operations to ensure
    /// that the basic building blocks are working correctly.
    mod nn {
        use super::*;

        #[test]
        fn test_forward_pass() {
            // A single hidden layer
            // let inputs: TensorRef = Tensor::new_random(vec![2, 3]);
            let inputs: TensorRef = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![2, 3]);
            let w1: TensorRef = Tensor::new(vec![1., 2., 3., 4., 5., 6.], vec![3, 2]);
            let b1: TensorRef = Tensor::new(vec![1., 1.], vec![2, 1]);

            let w2: TensorRef = Tensor::new(vec![1., 2., 3., 4.], vec![2, 2]);
            let b2: TensorRef = Tensor::new(vec![1., 1.], vec![2, 1]);

            let hidden1 = inputs.mm(&w1);
            let pre_activated1 = &hidden1 + &b1;
            let activated1 = pre_activated1.relu();

            let hidden2 = activated1.mm(&w2);
            let pre_activated2 = &hidden2 + &b2;
            let activated2 = pre_activated2.relu();
            // TODO: softmax needs a dim arg
            let output = activated2.softmax(1);
            let target = Tensor::new(vec![1., 0., 0., 1.], vec![2, 2]);

            // print all steps
            println!("hidden1: {:?}", hidden1);
            println!("pre_activated1: {:?}", pre_activated1);
            println!("activated1: {:?}", activated1);
            println!("hidden2: {:?}", hidden2);
            println!("pre_activated2: {:?}", pre_activated2);
            println!("activated2: {:?}", activated2);
            println!("output: {:?}", output);
            println!("target: {:?}", target);

            let ce = output.cross_entropy(&target);
            let loss = ce.mean(0);
            println!("cross_entropy: {:?}", ce);
            println!("loss: {:?}", loss);

            assert_eq!(output.shape, vec![2, 2]);
        }
    }

    mod broadcast {
        use super::*;

        #[test]
        fn test_elemwise_op_broadcasting() {
            // Test add sub mul div
            let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 2]);
            let b = Tensor::new(vec![5.0, 7.0], vec![1, 2, 1]);

            let add_result = Tensor::new(vec![6., 7., 8., 9., 8., 9., 10., 11.], vec![2, 2, 2]);

            let sub_result =
                Tensor::new(vec![-4., -3., -6., -5., -2., -1., -4., -3.], vec![2, 2, 2]);

            // [ 5, 10,  7, 14, 15, 20, 21, 28]
            let mul_result = Tensor::new(vec![5., 10., 7., 14., 15., 20., 21., 28.], vec![2, 2, 2]);
            // [0.2000, 0.4000, 0.1429, 0.2857, 0.6000, 0.8000, 0.4286, 0.5714]
            let div_result = Tensor::new(
                vec![0.2, 0.4, 0.14285, 0.28571, 0.6, 0.8, 0.42857, 0.57142],
                vec![2, 2, 2],
            );

            // assert
            assert_eq!(&a + &b, add_result);
            assert_eq!(&a - &b, sub_result);
            assert_eq!(&a * &b, mul_result);
            // for loop to compare
            let res = &a / &b;
            assert!(
                res.approx_eq(&div_result, 1e-5),
                "Expected: {:?}, Got: {:?}",
                div_result.data,
                res.data
            );
        }

        #[test]
        fn test_mm_broadcasting() {
            // Test mm
            let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 2]);
            let b = Tensor::new(vec![5.0, 7.0], vec![1, 2, 1]);

            let result = a.mm(&b);
            let expected = Tensor::new(vec![19., 43.], vec![2, 1, 1]);
            assert_eq!(result, expected);
            let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![1, 2, 2]); // broadcast batch dim
            let b = Tensor::new(vec![5.0, 6.0, 7.0, 8.0, 9.0, 10.0], vec![3, 2, 1]);

            let result = a.mm(&b);
            // Expected from torch
            let expected = Tensor::new(vec![17., 39., 23., 53., 29., 67.], vec![3, 2, 1]);
            assert_eq!(result, expected);
        }

        mod backwards {
            use super::*;

            #[test]
            fn test_add_broadcast_backwards() {
                // We repeat the tests above but test the gradients of the inital tensors
                let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 2]);
                let b = Tensor::new(vec![5.0, 7.0], vec![1, 2, 1]);

                let c = &a + &b;
                let d = c.sum(0);
                let e = d.sum(0);
                d.backward();
                // grad are refcelss of vecs
                let a_grad = a.grad.borrow();
                let a_grad = a_grad.as_ref().unwrap();
                let b_grad = b.grad.borrow();
                let b_grad = b_grad.as_ref().unwrap();
                assert_eq!(*a_grad, vec![2.0, 2.0, 2.0, 2.0]);
                assert_eq!(*b_grad, vec![4.0, 4.0]);
            }

            #[test]
            fn test_sub_broadcast_backwards() {
                // We repeat the tests above but test the gradients of the inital tensors
                let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 2]);
                let b = Tensor::new(vec![5.0, 7.0], vec![1, 2, 1]);

                let c = &a - &b;
                let d = c.sum(0);
                d.backward();
                let a_grad = a.grad.borrow();
                let a_grad = a_grad.as_ref().unwrap();
                let b_grad = b.grad.borrow();
                let b_grad = b_grad.as_ref().unwrap();
                assert_eq!(*a_grad, vec![2.0, 2.0, 2.0, 2.0]);
                assert_eq!(*b_grad, vec![-4.0, -4.0]);
            }

            #[test]
            fn test_mul_broadcast_backwards() {
                // We repeat the tests above but test the gradients of the inital tensors
                let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 2]);
                let b = Tensor::new(vec![5.0, 7.0], vec![1, 2, 1]);

                let c = &a * &b;
                let d = c.sum(0);
                d.backward();
                let a_grad = a.grad.borrow();
                let a_grad = a_grad.as_ref().unwrap();
                let b_grad = b.grad.borrow();
                let b_grad = b_grad.as_ref().unwrap();

                assert_eq!(*a_grad, vec![12.0, 12.0, 12.0, 12.0]);
                assert_eq!(*b_grad, vec![10.0, 10.0]);
            }

            #[test]
            fn test_div_broadcast_backwards() {
                // We repeat the tests above but test the gradients of the inital tensors
                let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 2]);
                let b = Tensor::new(vec![5.0, 7.0], vec![1, 2, 1]);

                let c = &a / &b;
                let d = c.sum(0);
                d.backward();
                let a_grad = a.grad.borrow();
                let a_grad = a_grad.as_ref().unwrap();
                let a_grad = Tensor::new(a_grad.clone(), a.shape.clone());
                let b_grad = b.grad.borrow();
                let b_grad = b_grad.as_ref().unwrap();
                let b_grad = Tensor::new(b_grad.clone(), b.shape.clone());
                assert!(
                    a_grad.approx_eq(
                        &Tensor::new(vec![0.3429, 0.3429, 0.3429, 0.3429], vec![2, 1, 2]),
                        1e-4
                    ),
                    "Expected: {:?}, Got: {:?}",
                    Tensor::new(vec![0.3429, 0.3429, 0.3429, 0.3429], vec![2, 1, 2]),
                    a_grad
                );
                assert!(
                    b_grad.approx_eq(&Tensor::new(vec![-0.4000, -0.2041], vec![1, 2, 1]), 1e-4),
                    "Expected: {:?}, Got: {:?}",
                    Tensor::new(vec![-0.4000, -0.2041], vec![1, 2, 1]),
                    b_grad
                );
            }

            // #[test]
            // fn test_mm_broadcast_backwards() {
            // test the batches mm
            // }
        }

        #[test]
        fn test_strides() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 2]);
            let b = Tensor::new(vec![5.0, 7.0], vec![1, 2, 1]);

            // Expected broadcasted shape: [2, 2, 2]
            let result = broadcast_shape(&a.shape, &b.shape);
            assert_eq!(result, vec![2, 2, 2]);

            // Check strides
            assert_eq!(a.strides, vec![2, 2, 1]);
            assert_eq!(b.strides, vec![2, 1, 1]);
        }

        #[test]
        fn test_shape_broadcasting() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], vec![2, 2, 2]);
            let b = Tensor::new(vec![5.0, 7.0], vec![1, 2, 1]);
            let result = broadcast_shape(&a.shape, &b.shape);
            assert_eq!(result, vec![2, 2, 2]);
        }

        #[test]
        fn test_broadcasting_iter_correctness() {
            let a = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 1, 2]);
            let b = Tensor::new(vec![5.0, 7.0], vec![1, 2, 1]);

            // Expected broadcasted shape: [2, 2, 2]
            let mut iter = BroadcastIterator::new(&a, &b);
            let mut count = 0;

            while let Some((i, j)) = iter.next() {
                let a_val = a.data[i];
                let b_val = b.data[j];
                assert!(a_val == 1.0 || a_val == 2.0 || a_val == 3.0 || a_val == 4.0);
                assert!(b_val == 5.0 || b_val == 7.0);

                count += 1;
            }

            // Expecting broadcast shape [2, 2, 2] => 8 iterations
            assert_eq!(count, 8);
        }
    }
}
