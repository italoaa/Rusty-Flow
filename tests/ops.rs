use rflow::tensor::Tensor;

#[cfg(test)]
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

    #[test]
    fn test_mse() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let b = Tensor::new(vec![4.0, 5.0, 6.0], vec![3]);
        let result = a.mse(&b);
        let expected = Tensor::new(vec![9.0, 9.0, 9.0], vec![3]);
        assert_eq!(result, expected);
    }

    #[test]
    fn test_ce() {
        let a = Tensor::new(vec![0.8, 0.1, 0.1], vec![3]);
        let b = Tensor::new(vec![0.0, 1.0, 0.0], vec![3]);
        let result = a.cross_entropy(&b);
        let expected = Tensor::new(vec![0.0, 1.60943791, 0.0], vec![3]);
        let mse = a.mse(&expected);
        for i in 0..mse.data.len() {
            assert!(
                mse.data[i] > 1e-6,
                "Expected {} to be close to {}",
                result.data[i],
                mse.data[i]
            );
        }
    }

    #[test]
    fn test_softmax() {
        let a = Tensor::new(vec![1.0, 2.0, 3.0], vec![3]);
        let result = a.softmax();
        let expected = Tensor::new(vec![0.0900, 0.2447, 0.6652], vec![3]);
        let mse = a.mse(&expected);
        for i in 0..result.data.len() {
            assert!(
                mse.data[i] > 1e-6,
                "Expected {} to be close to {}",
                result.data[i],
                expected.data[i]
            );
        }
    }
}
