use std::vec::Vec;

pub trait GradFn {
    fn backward(&self, grad_output: &Vec<f32>) -> Vec<Vec<f32>>;
}

struct SumBack {
    input_shape: Vec<usize>,
}

impl GradFn for SumBack {
    fn backward(&self, grad_output: &Vec<f32>) -> Vec<Vec<f32>> {
        // grad_output is scalar [1] (derivative of the loss wrt the sum)
        // Return a vector of ones with the same shape as input, times grad_output[0]
        let size = self.input_shape.iter().product();
        vec![vec![grad_output[0]; size]]
    }
}

struct Addback;
impl GradFn for Addback {
    fn backward(&self, grad_output: &Vec<f32>) -> Vec<Vec<f32>> {
        vec![grad_output.clone(), grad_output.clone()]
    }
}

impl<'a, 'b> Add<&'b TensorRef> for &'a TensorRef {
    type Output = TensorRef;

    fn add(self, other: &'b TensorRef) -> TensorRef {
        assert_eq!(self.shape, other.shape);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a + b)
            .collect();

        let requires_grad = self.requires_grad || other.requires_grad;

        // logging parents
        let parents = vec![Rc::downgrade(&self.0), Rc::downgrade(&other.0)];
        // println!("[add] New has parents: {:?}", parents);

        let grad_fn = if requires_grad {
            Some(Rc::new(Addback) as Rc<dyn GradFn>)
        } else {
            None
        };

        Tensor::new_with_options(data, self.shape.clone(), requires_grad, grad_fn, parents)
    }
}
