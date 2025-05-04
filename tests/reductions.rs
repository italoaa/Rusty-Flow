use rflow::tensor::Tensor;

#[cfg(test)]
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
