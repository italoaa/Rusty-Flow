use rflow::broadcast::{broadcast_shape, BroadcastIterator};
use rflow::tensor::Tensor;

#[cfg(test)]
mod others {
    use super::*;

    mod nn {
        use super::*;

        #[test]
        fn test_nn() {
            // 2 batches and 2 inputs per batch
            let inputs = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
            let weights = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
            let biases = Tensor::new(vec![1.0, 1.0, 1.0, 1.0], vec![2, 2]);

            let preacts = &inputs.mm(&weights) + &biases;
            let activations = preacts.softmax();
            let _expected_activations = Tensor::new(vec![1.0, 2.0, 3.0, 4.0], vec![2, 2]);
            // assert_eq!(activations, expected_activations);

            // Targets one hot
            let targets = Tensor::new(vec![0.0, 1.0, 1.0, 0.0], vec![2, 2]);

            // Cross entropy loss
            let loss = activations.cross_entropy(&targets);
            // assert_eq!(loss, Tensor::new(vec![], vec![2, 2]));

            loss.backward();

            let weight_grad = weights.grad.borrow();
            let _weight_grad = weight_grad.as_ref().unwrap();
            // assert_eq!(weight_grad, &vec![0.0, 0.0, 0.0, 0.0]);

            let bias_grad = biases.grad.borrow();
            let _bias_grad = bias_grad.as_ref().unwrap();
            // assert_eq!(bias_grad, &vec![0.0, 0.0, 0.0, 0.0]);
        }
    }
    mod broadcast {
        use super::*;

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

                // Since a has shape [2,1,2], its data is arranged as:
                // [
                //   [[1.0, 2.0]],
                //   [[3.0, 4.0]]
                // ]
                // b has shape [1,2,1], data: [[5.0], [7.0]]
                // broadcast shape = [2,2,2]

                // For each of the 8 combinations, we can precompute what the correct a_val, b_val should be.
                // You can also store expected pairs here and assert.
                // println!("a_val: {}, b_val: {}, i: {}, j: {}", a_val, b_val, i, j);

                // Optionally: assert broadcasting behavior
                assert!(a_val == 1.0 || a_val == 2.0 || a_val == 3.0 || a_val == 4.0);
                assert!(b_val == 5.0 || b_val == 7.0);

                count += 1;
            }

            // Expecting broadcast shape [2, 2, 2] => 8 iterations
            assert_eq!(count, 8);
        }
    }
}
