use crate::is_debug;
use crate::tensor::{Tensor, TensorRef};
use crate::utils::topo_sort;
use std::rc::Rc;

// === grad fn ===

/// Gradient river flow starts at the loss value and flows DOWNwards. As such the gradient from the childern nodes is the upstream gradient and the gradient to the parents is the downstream gradient.
/// Compute gradients w.r.t. each input, given the gradient w.r.t. this op’s output.
pub trait GradFn {
    /// # Arguments
    ///
    /// * `upstream_grad` – the gradient of the loss with respect to this op’s output
    ///
    /// # Returns
    ///
    /// A `Vec` of gradients—one `Vec<f32>` per parent tensor—each matching that parent’s shape.
    fn backward(&self, upstream_grad: &Vec<f32>) -> Vec<Vec<f32>>;
}

// ================ SUM OPERATION ==================
pub struct SumBack {
    pub input_shape: Vec<usize>,
}

impl GradFn for SumBack {
    fn backward(&self, upstream_grad: &Vec<f32>) -> Vec<Vec<f32>> {
        // upstream_grad is scalar [1] (derivative of the loss wrt the sum)
        // Return a vector of ones with the same shape as input, times upstream_grad[0]
        let size = self.input_shape.iter().product();
        vec![vec![upstream_grad[0]; size]]
    }
}

// ================ ADD OPERATION ==================

pub struct AddBack;
impl GradFn for AddBack {
    fn backward(&self, upstream_grad: &Vec<f32>) -> Vec<Vec<f32>> {
        vec![upstream_grad.clone(), upstream_grad.clone()]
    }
}

// ================ MatMul OPERATION ==================

pub struct MMBack {
    pub left: Rc<Tensor>,
    pub right: Rc<Tensor>,
}

impl GradFn for MMBack {
    fn backward(&self, upstream_grad: &Vec<f32>) -> Vec<Vec<f32>> {
        // The gradient of the matrix multiplication is a bit more complex
        // f(x) = x @ y
        // df/dx = y^T
        // df/dy = x^T
        // dL/dx = dL/dthis * y^T
        // dL/dy = x^T * dL/dthis

        // left is m, n
        // right is n, p
        // upstream_grad is m, p

        let m = self.left.shape[0];
        let p = self.right.shape[1];
        let right = self.right.transpose();
        let left = self.left.transpose();

        let grads_tensor = Tensor::new(upstream_grad.clone(), vec![m, p]);

        let grad_left = grads_tensor.mm(&right);
        let grad_right = left.mm(&grads_tensor);

        return vec![grad_left.data.clone(), grad_right.data.clone()];
    }
}

// ================ SUBSTRACT OPERATION ==================

pub struct SubBack;
impl GradFn for SubBack {
    fn backward(&self, upstream_grad: &Vec<f32>) -> Vec<Vec<f32>> {
        // gradient of a subtraction is the same as addition,
        // but the second input is negated
        // f(x) = x - y
        // df/dx = 1
        // df/dy = -1
        vec![
            upstream_grad.clone(),
            upstream_grad.iter().map(|x| -x).collect(),
        ]
    }
}

// ================ MULTIPLY OPERATION ==================

pub struct MulBack {
    pub left: Rc<Tensor>,
    pub right: Rc<Tensor>,
}
impl GradFn for MulBack {
    fn backward(&self, upstream_grad: &Vec<f32>) -> Vec<Vec<f32>> {
        // println!("[Mulback] Backward called with upstream_grad: {:?}", upstream_grad);
        // The gradient of the product is taking the output gradient times the other input
        // f(x) = x * y
        // df/dx = y
        // df/dy = x

        // grad left takes the right data
        let grad_left: Vec<f32> = upstream_grad
            .iter()
            .zip(self.right.data.iter())
            .map(|(grad, rightdata)| grad * rightdata)
            .collect();

        // grad right takes the left data
        let grad_right: Vec<f32> = upstream_grad
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
    fn backward(&self, upstream_grad: &Vec<f32>) -> Vec<Vec<f32>> {
        // The gradient of the division is taking
        // the output gradient times the other input

        // f(x) = x / y
        // df/dx = 1/y
        // df/dy = -x/y^2

        let grad_left: Vec<f32> = upstream_grad
            .iter()
            .zip(self.right.data.iter())
            .map(|(grad, rightdata)| grad / rightdata)
            .collect();

        let grad_right: Vec<f32> = upstream_grad
            .iter()
            .zip(self.left.data.iter())
            .zip(self.right.data.iter())
            .map(|((grad, l), r)| -grad * l / (r * r))
            .collect();

        vec![grad_left, grad_right]
    }
}

// ================ RELU ==================

pub struct ReLUBack {
    pub input: Rc<Tensor>,
}

impl GradFn for ReLUBack {
    fn backward(&self, upstream_grad: &Vec<f32>) -> Vec<Vec<f32>> {
        // The gradient of ReLU is 1 for positive inputs and 0 for negative inputs
        let grad_input: Vec<f32> = self
            .input
            .data
            .iter()
            .zip(upstream_grad.iter())
            .map(|(input, grad)| if *input > 0.0 { *grad } else { 0.0 })
            .collect();

        vec![grad_input]
    }
}

// ================ MSE ==================

pub struct MSEBack {
    pub input: Rc<Tensor>,
    pub target: Rc<Tensor>,
}

impl GradFn for MSEBack {
    fn backward(&self, upstream_grad: &Vec<f32>) -> Vec<Vec<f32>> {
        // The gradient of the MSE loss is 2 * (input - target)
        let grad_input: Vec<f32> = self
            .input
            .data
            .iter()
            .zip(self.target.data.iter())
            .zip(upstream_grad.iter())
            .map(|((input, target), grad)| 2.0 * (*input - *target) * *grad)
            .collect();

        vec![grad_input]
    }
}

// ================ CrossEntropy ==================

pub struct CrossEntropyBack {
    pub input: Rc<Tensor>,
    pub target: Rc<Tensor>,
}

impl GradFn for CrossEntropyBack {
    fn backward(&self, upstream_grad: &Vec<f32>) -> Vec<Vec<f32>> {
        // The gradient of the cross-entropy loss is -target / input
        let grad_input: Vec<f32> = self
            .input
            .data
            .iter()
            .zip(self.target.data.iter())
            .zip(upstream_grad.iter())
            .map(|((input, target), grad)| -(*target / *input) * *grad)
            .collect();

        vec![grad_input]
    }
}

// ================ SOFTMAX ==================
pub struct SoftmaxBack {
    pub output: Rc<Tensor>,
}

impl GradFn for SoftmaxBack {
    fn backward(&self, upstream_grad: &Vec<f32>) -> Vec<Vec<f32>> {
        // grad
        // softmax(x)
        // dL/dxi = SUM_j(dL/dsj * dsj/dxi)
        // we can simplify this to
        // dL/dxi = si * (dL/dsi - SUM_j(dL/dsj * sj))
        let s = &self.output.data; // softmax output

        // Compute dot(s, upstream_grad)
        let dot = s
            .iter()
            .zip(upstream_grad.iter())
            .map(|(si, gi)| si * gi)
            .sum::<f32>();

        let grad_input: Vec<f32> = s
            .iter()
            .zip(upstream_grad.iter())
            .map(|(si, gi)| si * (gi - dot))
            .collect();

        vec![grad_input]
    }
}

// ================ CE + SOFTMAX ==================

pub struct CrossEntropyLogitsBack {
    pub softmax: Rc<Tensor>,
    pub target: Rc<Tensor>,
}

impl GradFn for CrossEntropyLogitsBack {
    fn backward(&self, _upstream_grad: &Vec<f32>) -> Vec<Vec<f32>> {
        // The gradient of the cross-entropy loss with softmax is
        // dL/dxi = softmax(xi) - target

        let downstream_grad: Vec<f32> = self
            .softmax
            .data
            .iter()
            .zip(self.target.data.iter())
            .map(|(softmax, target)| softmax - target)
            .collect();

        vec![downstream_grad]
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

                // grad is the upstream gradient
                let parent_grads = grad_fn.backward(grad);
                for (parent_weak, parent_new_grad) in node.parents.iter().zip(parent_grads.iter()) {
                    if is_debug() {
                        println!(
                            "[autodiff]   - Parent: {:?} with grad {:?}",
                            parent_weak, parent_new_grad
                        );
                    }
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
                                if is_debug() {
                                    println!("[autodiff]   - Existing grad: {:?}", existing);
                                }
                                for (e, n) in existing.iter_mut().zip(parent_new_grad) {
                                    *e += n;
                                }
                            }
                            None => {
                                if is_debug() {
                                    println!(
                                        "[autodiff]   - Setting new grad: {:?}",
                                        parent_new_grad
                                    );
                                }
                                parent_grad.replace(parent_new_grad.clone());
                            }
                        }
                    } else {
                        panic!("[autodiff] Parent weak reference is None");
                    }
                }
            }
        }
    }
}
