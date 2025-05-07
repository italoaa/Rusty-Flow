use rflow::optimizers::SGD;
use rflow::tensor::Tensor;

#[cfg(test)]
mod backward {
    use super::*;

    #[test]
    fn test_sgd_lr() {
        let a = Tensor::new(vec![1., 2., 3.], vec![3]);
        a.set_grad(vec![2.0, 1.0, -4.0]);

        let mut optim = SGD::new(vec![a.rc()], 0.01, 0.0, 0.);

        optim.step();

        let expected = Tensor::new(vec![0.98, 1.99, 3.04], vec![3]);
        assert!(
            a.approx_eq(&expected, 1e-5),
            "Expected: {:?}, Actual: {:?}",
            expected,
            a
        );
    }

    #[test]
    fn test_sgd_momentum() {
        let a = Tensor::new(vec![1., 2., 3.], vec![3]);
        a.set_grad(vec![2.0, 1.0, -4.0]);

        let mut optim = SGD::new(vec![a.rc()], 0.01, 0.9, 0.);

        optim.step();
        let expected = Tensor::new(vec![0.98, 1.99, 3.04], vec![3]);
        assert!(
            a.approx_eq(&expected, 1e-5),
            "Expected: {:?}, Actual: {:?}",
            expected,
            a
        );

        a.set_grad(vec![-2.0, -1.0, 4.0]);
        optim.step();
        let expected = Tensor::new(vec![0.9820, 1.9910, 3.0360], vec![3]);
        assert!(
            a.approx_eq(&expected, 1e-5),
            "Expected: {:?}, Actual: {:?}",
            expected,
            a
        );
    }

    #[test]
    fn test_sgd_wd() {
        let a = Tensor::new(vec![1., 2., 3.], vec![3]);
        a.set_grad(vec![2.0, 1.0, -4.0]);

        // extreme wd to test
        let mut optim = SGD::new(vec![a.rc()], 0.01, 0.0, 10.0);

        optim.step();

        let expected = Tensor::new(vec![0.8800, 1.7900, 2.7400], vec![3]);
        assert!(
            a.approx_eq(&expected, 1e-5),
            "Expected: {:?}, Actual: {:?}",
            expected,
            a
        );
    }

    #[test]
    fn test_sgd_zero_grad() {
        let a = Tensor::new(vec![1., 2., 3.], vec![3]);
        a.set_grad(vec![2.0, 1.0, -4.0]);

        let mut optim = SGD::new(vec![a.rc()], 0.01, 0.0, 0.);
        optim.zero_grad();
        let expected = Tensor::new(vec![1., 2., 3.], vec![3]);
        expected.set_grad(vec![0.0, 0.0, 0.0]);

        assert!(
            a.approx_eq(&expected, 1e-5),
            "Expected: {:?}, Actual: {:?}",
            expected,
            a
        );
    }
}
