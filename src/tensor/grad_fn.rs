use crate::tensor::tensor::Tensor;
use crate::tensor::tensor_ref::TensorRef;
use std::{
    cell::RefCell,
    collections::HashSet,
    fmt,
    ops::{Add, Deref, Div, Mul, Sub},
    rc::{Rc, Weak},
};

pub trait GradFn {
    fn backward(&self, grad_output: &Vec<f32>) -> Vec<Vec<f32>>;
}

pub struct SumBack {
    pub input_shape: Vec<usize>,
}

impl GradFn for SumBack {
    fn backward(&self, grad_output: &Vec<f32>) -> Vec<Vec<f32>> {
        // grad_output is scalar [1] (derivative of the loss wrt the sum)
        // Return a vector of ones with the same shape as input, times grad_output[0]
        let size = self.input_shape.iter().product();
        vec![vec![grad_output[0]; size]]
    }
}

pub struct MMback {
    pub left: TensorRef,
    pub right: TensorRef,
}
impl GradFn for MMback {
    fn backward(&self, grad_output: &Vec<f32>) -> Vec<Vec<f32>> {
        // The gradient of the matrix multiplication is a bit more complex
        // f(x) = x @ y
        // df/dx = y^T
        // df/dy = x^T
        // dL/dx = dL/dthis * y^T
        // dL/dy = x^T * dL/dthis

        // left is m, n
        // right is n, p
        // grad_output is m, p

        let m = self.left.shape[0];
        let p = self.right.shape[1];
        let right = self.right.transpose();
        let left = self.left.transpose();

        let grads_tensor = Tensor::new(grad_output.clone(), vec![m, p]);

        let grad_left = grads_tensor.matmul(&right);
        let grad_right = left.matmul(&grads_tensor);

        return vec![grad_left.data.clone(), grad_right.data.clone()];
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

struct Subback;
impl GradFn for Subback {
    fn backward(&self, grad_output: &Vec<f32>) -> Vec<Vec<f32>> {
        // gradient of a subtraction is the same as addition,
        // but the second input is negated
        // f(x) = x - y
        // df/dx = 1
        // df/dy = -1
        vec![
            grad_output.clone(),
            grad_output.iter().map(|x| -x).collect(),
        ]
    }
}

impl<'a, 'b> Sub<&'b TensorRef> for &'a TensorRef {
    type Output = TensorRef;

    fn sub(self, other: &'b TensorRef) -> TensorRef {
        assert_eq!(self.shape, other.shape);
        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a - b)
            .collect();

        let requires_grad = self.requires_grad || other.requires_grad;

        let grad_fn = if requires_grad {
            Some(Rc::new(Subback) as Rc<dyn GradFn>)
        } else {
            None
        };

        let parents = vec![Rc::downgrade(&self.0), Rc::downgrade(&other.0)];

        Tensor::new_with_options(data, self.shape.clone(), requires_grad, grad_fn, parents)
    }
}

struct Mulback {
    left: Rc<Tensor>,
    right: Rc<Tensor>,
}
impl GradFn for Mulback {
    fn backward(&self, grad_output: &Vec<f32>) -> Vec<Vec<f32>> {
        // println!("[Mulback] Backward called with grad_output: {:?}", grad_output);
        // The gradient of the product is taking the output gradient times the other input
        // f(x) = x * y
        // df/dx = y
        // df/dy = x

        // grad left takes the right data
        let grad_left: Vec<f32> = grad_output
            .iter()
            .zip(self.right.data.iter())
            .map(|(grad, rightdata)| grad * rightdata)
            .collect();

        // grad right takes the left data
        let grad_right: Vec<f32> = grad_output
            .iter()
            .zip(self.left.data.iter())
            .map(|(grad, leftdata)| grad * leftdata)
            .collect();

        vec![grad_left, grad_right]
    }
}

impl<'a, 'b> Mul<&'b TensorRef> for &'a TensorRef {
    type Output = TensorRef;

    fn mul(self, other: &'b TensorRef) -> TensorRef {
        assert_eq!(
            self.shape, other.shape,
            "Shape mismatch for elementwise multiplication"
        );

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a * b)
            .collect::<Vec<_>>();

        let requires_grad = self.requires_grad || other.requires_grad;

        let grad_fn = if requires_grad {
            Some(Rc::new(Mulback {
                left: self.0.clone(),
                right: other.0.clone(),
            }) as Rc<dyn GradFn>)
        } else {
            None
        };

        let parents = vec![Rc::downgrade(&self.0), Rc::downgrade(&other.0)];

        Tensor::new_with_options(data, self.shape.clone(), requires_grad, grad_fn, parents)
    }
}

struct Divback {
    left: Rc<Tensor>,
    right: Rc<Tensor>,
}
impl GradFn for Divback {
    fn backward(&self, grad_output: &Vec<f32>) -> Vec<Vec<f32>> {
        // The gradient of the division is taking
        // the output gradient times the other input

        // f(x) = x / y
        // df/dx = 1/y
        // df/dy = -x/y^2

        let grad_left: Vec<f32> = grad_output
            .iter()
            .zip(self.right.data.iter())
            .map(|(grad, rightdata)| grad / rightdata)
            .collect();

        let grad_right: Vec<f32> = grad_output
            .iter()
            .zip(self.left.data.iter())
            .zip(self.right.data.iter())
            .map(|((grad, l), r)| -grad * l / (r * r))
            .collect();

        vec![grad_left, grad_right]
    }
}

impl<'a, 'b> Div<&'b TensorRef> for &'a TensorRef {
    type Output = TensorRef;

    fn div(self, other: &'b TensorRef) -> TensorRef {
        assert_eq!(
            self.shape, other.shape,
            "Shape mismatch for elementwise division"
        );

        let data = self
            .data
            .iter()
            .zip(other.data.iter())
            .map(|(a, b)| a / b)
            .collect::<Vec<_>>();

        let requires_grad = self.requires_grad || other.requires_grad;

        let grad_fn = if requires_grad {
            Some(Rc::new(Divback {
                left: self.0.clone(),
                right: other.0.clone(),
            }) as Rc<dyn GradFn>)
        } else {
            None
        };

        let parents = vec![Rc::downgrade(&self.0), Rc::downgrade(&other.0)];

        Tensor::new_with_options(data, self.shape.clone(), requires_grad, grad_fn, parents)
    }
}
