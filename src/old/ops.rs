use crate::tensor::grad_fn::SumBack;
use crate::tensor::{GradFn, Tensor};

impl TensorRef {
    pub fn requires_grad(&self) -> bool {
        self.0.requires_grad
    }

    pub fn sum(&self) -> TensorRef {
        let sum_val = self.data.iter().sum();
        // If no grads needed, just return plain leaf tensor
        if !self.requires_grad {
            return Tensor::new(vec![sum_val], vec![1]);
        }

        let grad_fn = Some(Rc::new(SumBack {
            input_shape: self.shape.clone(),
        }) as Rc<dyn GradFn>);
        let parents = vec![Rc::downgrade(&self.0)];

        // Construct a scalar tensor with grad_fn and parents
        Tensor::new_with_options(vec![sum_val], vec![], true, grad_fn, parents)
    }

    pub fn mean(&self) -> TensorRef {
        let sum: TensorRef = self.sum();
        // Divide by the number of elements
        let len: usize = self.shape.iter().product();
        // no broadcasting
        let div = Tensor::new(vec![len as f32], vec![]);
        &sum / &div
    }

    pub fn matmul(&self, other: &TensorRef) -> TensorRef {
        // TODO: impl broadcasting
        assert_eq!(
            self.shape[1], other.shape[0],
            "Incompatible shapes for matrix multiplication"
        );

        let mut result = vec![0.0; self.shape[0] * other.shape[1]];

        // assumes row-major order
        for i in 0..self.shape[0] {
            for j in 0..other.shape[1] {
                for k in 0..self.shape[1] {
                    result[i * other.shape[1] + j] +=
                        self.data[i * self.shape[1] + k] * other.data[k * other.shape[1] + j];
                }
            }
        }

        Tensor::new(result, vec![self.shape[0], other.shape[1]])
    }
}
