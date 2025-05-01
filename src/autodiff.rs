use crate::is_debug;
use crate::tensor::{Tensor, TensorRef};
use crate::utils::topo_sort;
use std::rc::Rc;

// === grad fn ===

pub trait GradFn {
    fn backward(&self, grad_output: &Vec<f32>) -> Vec<Vec<f32>>;
}

// ================ SUM OPERATION ==================
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

// ================ ADD OPERATION ==================

pub struct AddBack;
impl GradFn for AddBack {
    fn backward(&self, grad_output: &Vec<f32>) -> Vec<Vec<f32>> {
        vec![grad_output.clone(), grad_output.clone()]
    }
}

// ================ MatMul OPERATION ==================

pub struct MMBack {
    pub left: Rc<Tensor>,
    pub right: Rc<Tensor>,
}

impl GradFn for MMBack {
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

        let grad_left = grads_tensor.mm(&right);
        let grad_right = left.mm(&grads_tensor);

        return vec![grad_left.data.clone(), grad_right.data.clone()];
    }
}

// ================ SUBSTRACT OPERATION ==================

pub struct SubBack;
impl GradFn for SubBack {
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

// ================ MULTIPLY OPERATION ==================

pub struct MulBack {
    pub left: Rc<Tensor>,
    pub right: Rc<Tensor>,
}
impl GradFn for MulBack {
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

// ================ DIVIDE OPERATION ==================

pub struct DivBack {
    pub left: Rc<Tensor>,
    pub right: Rc<Tensor>,
}
impl GradFn for DivBack {
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

// ================ Backwards ========================

pub fn backward(tensor: &TensorRef) {
    assert!(
        tensor.requires_grad,
        "[autodiff] Tensor does not require grad"
    );
    assert!(tensor.shape == vec![], "[autodiff] Tensor is not a scalar");
    {
        let mut grad_ref = tensor.grad.borrow_mut();
        if grad_ref.is_none() {
            grad_ref.replace(vec![1.0]);
        } else {
            panic!("[autodiff] Output grad already set");
        }
    }

    if is_debug() {
        println!("[autodiff] Starting backward pass at node {:?}", tensor);
    }

    let sorted = topo_sort(&tensor.0);
    for node in sorted.into_iter().rev() {
        if is_debug() {
            println!("[autodiff] Processing node {:?}", node); // assuming you have an id or name
        }

        if let Some(grad_fn) = &node.grad_fn {
            if let Some(grad) = node.grad.borrow().as_ref() {
                if is_debug() {
                    println!("[autodiff] - Grad: {:?}", grad);
                }

                let parent_grads = grad_fn.backward(grad);
                for (parent_weak, parent_new_grad) in node.parents.iter().zip(parent_grads.iter()) {
                    if let Some(parent_rc) = parent_weak.upgrade() {
                        if is_debug() {
                            println!(
                                "[autodiff]   - Propagating to parent {:?} with grad {:?}",
                                parent_rc, parent_new_grad
                            );
                        }
                        let mut parent_grad = parent_rc.grad.borrow_mut();
                        match &mut *parent_grad {
                            Some(existing) => {
                                for (e, n) in existing.iter_mut().zip(parent_new_grad) {
                                    *e += n;
                                }
                            }
                            None => {
                                parent_grad.replace(parent_new_grad.clone());
                            }
                        }
                    }
                }
            }
        }
    }
}
