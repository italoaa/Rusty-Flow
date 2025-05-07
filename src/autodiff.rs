use crate::broadcast::{broadcast_shape, resolve_broadcast_index, BroadcastIterator};
use crate::is_debug;
use crate::tensor::{Tensor, TensorRef};
use crate::utils::slices_along_dim;
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

fn unbroadcast_grads(
    upstream_grad: &Vec<f32>,
    out_shape: &Vec<usize>,
    in_shape: &Vec<usize>,
) -> Vec<f32> {
    let mut grad = vec![0.0; in_shape.iter().product()];
    for i in 0..upstream_grad.len() {
        let j = resolve_broadcast_index(i, out_shape, in_shape);
        grad[j] += upstream_grad[i];
    }
    grad
}

// ================ SUM OPERATION ==================
pub struct SumBack {
    pub input_shape: Vec<usize>,
    pub dim: usize,
}

impl GradFn for SumBack {
    fn backward(&self, upstream_grad: &Vec<f32>) -> Vec<Vec<f32>> {
        // upstream_grad is scalar [1] for each slice (derivative of the loss wrt the sum)
        let mut grad = vec![0.0; self.input_shape.iter().product()];

        for (i, slice) in slices_along_dim(&self.input_shape, self.dim).enumerate() {
            // the slice is a vec of flat indicies
            for j in 0..slice.len() {
                if grad[slice[j]] != 0.0 {
                    panic!("[autodiff] This grad should not be set, the slicing of the tensor is not correct: data: {:?}, slice: {:?}", self.input_shape, slice);
                }
                grad[slice[j]] = upstream_grad[i];
            }
        }

        // No need to unbroadcast the gradient, since we are already in the input space
        vec![grad]
    }
}

// ================ MEAN OPERATION ==================
pub struct MeanBack {
    pub input_shape: Vec<usize>,
    pub dim: usize,
}

impl GradFn for MeanBack {
    fn backward(&self, upstream_grad: &Vec<f32>) -> Vec<Vec<f32>> {
        // upstream_grad is scalar [1] for each slice (derivative of the loss wrt the mean)
        let mut grad = vec![0.0; self.input_shape.iter().product()];

        // Number of elements along the specified dimension
        let num_elements = self.input_shape[self.dim] as f32;

        // Scale the upstream gradient by 1 / num_elements
        let scale = 1.0 / num_elements;

        // Iterate over slices along the specified dimension
        for (i, slice) in slices_along_dim(&self.input_shape, self.dim).enumerate() {
            for j in 0..slice.len() {
                if grad[slice[j]] != 0.0 {
                    panic!("[autodiff] This grad should not be set, the slicing of the tensor is not correct: data: {:?}, slice: {:?}", self.input_shape, slice);
                }
                // Propagate the scaled upstream gradient
                grad[slice[j]] = upstream_grad[i] * scale;
            }
        }

        // No need to unbroadcast the gradient, since we are already in the input space
        vec![grad]
    }
}

// ================ ADD OPERATION ==================

pub struct AddBack {
    pub left_shape: Vec<usize>,
    pub right_shape: Vec<usize>,
    pub out_shape: Vec<usize>,
}
impl GradFn for AddBack {
    fn backward(&self, upstream_grad: &Vec<f32>) -> Vec<Vec<f32>> {
        let left_grad = unbroadcast_grads(upstream_grad, &self.out_shape, &self.left_shape);
        let right_grad = unbroadcast_grads(upstream_grad, &self.out_shape, &self.right_shape);

        vec![left_grad, right_grad]
    }
}

// ================ SUBSTRACT OPERATION ==================

pub struct SubBack {
    pub left_shape: Vec<usize>,
    pub right_shape: Vec<usize>,
    pub out_shape: Vec<usize>,
}
impl GradFn for SubBack {
    fn backward(&self, upstream_grad: &Vec<f32>) -> Vec<Vec<f32>> {
        // gradient of a subtraction is the same as addition,
        // but the second input is negated
        // f(x) = x - y
        // df/dx = 1
        // df/dy = -1
        let left_grad = unbroadcast_grads(upstream_grad, &self.out_shape, &self.left_shape);
        let right_grad = unbroadcast_grads(upstream_grad, &self.out_shape, &self.right_shape);

        vec![left_grad, right_grad.iter().map(|x| -x).collect()]
    }
}

// ================ MULTIPLY OPERATION ==================

pub struct MulBack {
    pub left: Rc<Tensor>,
    pub right: Rc<Tensor>,
    pub out_shape: Vec<usize>,
}
impl GradFn for MulBack {
    fn backward(&self, upstream_grad: &Vec<f32>) -> Vec<Vec<f32>> {
        // alkdjfa
        // we first multiply the upstream gradient by the other input
        let mut left_grad = Vec::new();
        let mut iter = BroadcastIterator::new_with_shapes(&self.out_shape, &self.right.shape);
        while let Some((i, j)) = iter.next() {
            left_grad.push(upstream_grad[i] * self.right.data.borrow()[j]);
        }

        let mut right_grad = Vec::new();
        let mut iter = BroadcastIterator::new_with_shapes(&self.out_shape, &self.left.shape);
        while let Some((i, j)) = iter.next() {
            right_grad.push(upstream_grad[i] * self.left.data.borrow()[j]);
        }

        // Unbroadcast the gradients
        let left_grad = unbroadcast_grads(&left_grad, &self.out_shape, &self.left.shape);

        let right_grad = unbroadcast_grads(&right_grad, &self.out_shape, &self.right.shape);

        // Return the gradients
        vec![left_grad, right_grad]
    }
}

// ================ DIVIDE OPERATION ==================

pub struct DivBack {
    pub left: Rc<Tensor>,
    pub right: Rc<Tensor>,
    pub out_shape: Vec<usize>,
}
impl GradFn for DivBack {
    fn backward(&self, upstream_grad: &Vec<f32>) -> Vec<Vec<f32>> {
        // The gradient of the division is taking
        // the output gradient times the other input

        // f(x) = x / y
        // df/dx = 1/y
        // df/dy = -x/y^2
        // First we do the operation broadcasted
        let mut left_grad = Vec::new();
        let mut iter = BroadcastIterator::new_with_shapes(&self.out_shape, &self.right.shape);
        while let Some((i, j)) = iter.next() {
            left_grad.push(upstream_grad[i] / self.right.data.borrow()[j]);
        }

        let mut right_grad = Vec::new();
        let mut iter_l = BroadcastIterator::new_with_shapes(&self.out_shape, &self.left.shape);
        let mut iter_r = BroadcastIterator::new_with_shapes(&self.out_shape, &self.right.shape);

        while let (Some((i_l, j_l)), Some((_, j_r))) = (iter_l.next(), iter_r.next()) {
            let l = self.left.data.borrow()[j_l];
            let r = self.right.data.borrow()[j_r];
            right_grad.push(-upstream_grad[i_l] * l / (r * r));
        }

        // Unbroadcast the gradients
        let left_grad = unbroadcast_grads(&left_grad, &self.out_shape, &self.left.shape);
        let right_grad = unbroadcast_grads(&right_grad, &self.out_shape, &self.right.shape);

        // Return the gradients
        vec![left_grad, right_grad]
    }
}

// ================ MatMul OPERATION ==================

pub struct MMBack {
    pub left: Rc<Tensor>,
    pub right: Rc<Tensor>,
    pub out_shape: Vec<usize>,
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
        let m = self.left.shape[self.left.shape.len() - 2];
        let n = self.left.shape[self.left.shape.len() - 1];
        let p = self.right.shape[self.right.shape.len() - 1];

        let left_batch_shape = &self.left.shape[..self.left.shape.len() - 2];
        let right_batch_shape = &self.right.shape[..self.right.shape.len() - 2];
        let out_batch_shape = &self.out_shape[..self.out_shape.len() - 2];

        // Create broadcast iterators for left and right tensors
        let mut iter_l = BroadcastIterator::new_with_shapes(&out_batch_shape, &left_batch_shape);
        let mut iter_r = BroadcastIterator::new_with_shapes(&out_batch_shape, &right_batch_shape);

        // Results
        let mut left_grad_accum = vec![0.0; self.left.data.borrow().len()];
        let mut right_grad_accum = vec![0.0; self.right.data.borrow().len()];

        // Iterate over the batch dimensions
        while let (Some((i_out, i_left)), Some((_, i_right))) = (iter_l.next(), iter_r.next()) {
            // Batch offsets
            let left_batch_offset = i_left * m * n;
            let right_batch_offset = i_right * n * p;
            let out_batch_offset = i_out * m * p;

            let upstream_grad_batch = &upstream_grad[out_batch_offset..out_batch_offset + m * p];
            let left_batch = &self.left.data.borrow()[left_batch_offset..left_batch_offset + m * n];
            let right_batch =
                &self.right.data.borrow()[right_batch_offset..right_batch_offset + n * p];

            let upstream_grad_tensor = Tensor::new(upstream_grad_batch.to_vec(), vec![m, p]);
            let left_batch_tensor = Tensor::new(left_batch.to_vec(), vec![m, n]);
            let right_batch_tensor = Tensor::new(right_batch.to_vec(), vec![n, p]);

            // Perform backprop for a single example
            let left_grad_batch = upstream_grad_tensor.mm(&right_batch_tensor.transpose());
            let right_grad_batch = left_batch_tensor.transpose().mm(&upstream_grad_tensor);

            // We are still in broadcasted space, so we need to unbroadcast
            let left_grad = unbroadcast_grads(
                &left_grad_batch.data.borrow(),
                &self.out_shape,
                &self.left.shape,
            );
            let right_grad = unbroadcast_grads(
                &right_grad_batch.data.borrow(),
                &self.out_shape,
                &self.right.shape,
            );

            // Add the gradients to the output
            for i in 0..left_grad.len() {
                left_grad_accum[i + left_batch_offset] += left_grad[i];
            }
            for i in 0..right_grad.len() {
                right_grad_accum[i + right_batch_offset] += right_grad[i];
            }
        }

        vec![left_grad_accum, right_grad_accum]
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
            .borrow()
            .iter()
            .zip(upstream_grad.iter())
            .map(|(input, grad)| if *input > 0.0 { *grad } else { 0.0 })
            .collect();

        vec![grad_input]
    }
}
// LRELU

pub struct LReLUBack {
    pub input: Rc<Tensor>,
    pub alpha: f32,
}

impl GradFn for LReLUBack {
    fn backward(&self, upstream_grad: &Vec<f32>) -> Vec<Vec<f32>> {
        // The gradient of ReLU is 1 for positive inputs and 0 for negative inputs
        let grad_input: Vec<f32> = self
            .input
            .data
            .borrow()
            .iter()
            .zip(upstream_grad.iter())
            .map(|(input, grad)| {
                if *input > 0.0 {
                    *grad
                } else {
                    self.alpha * *grad
                }
            })
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
            .borrow()
            .iter()
            .zip(self.target.data.borrow().iter())
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
            .borrow()
            .iter()
            .zip(self.target.data.borrow().iter())
            .zip(upstream_grad.iter())
            .map(|((input, target), grad)| -(*target / *input) * *grad)
            .collect();

        vec![grad_input]
    }
}

// ================ SOFTMAX ==================
pub struct SoftmaxBack {
    pub output: Rc<Tensor>,
    pub dim: usize,
}

impl GradFn for SoftmaxBack {
    fn backward(&self, upstream_grad: &Vec<f32>) -> Vec<Vec<f32>> {
        // grad
        // softmax(x)
        // dL/dxi = SUM_j(dL/dsj * dsj/dxi)
        // we can simplify this to
        // dL/dxi = si * (dL/dsi - SUM_j(dL/dsj * sj))

        let s = &self.output.data;
        let mut grad_input = vec![0.0; s.borrow().len()];

        for indices in self.output.iterate_over_dim_indices(self.dim) {
            let s_slice: Vec<f32> = indices.iter().map(|&i| s.borrow()[i]).collect();
            let upstream_slice: Vec<f32> = indices.iter().map(|&i| upstream_grad[i]).collect();

            let dot: f32 = s_slice
                .iter()
                .zip(upstream_slice.iter())
                .map(|(si, gi)| si * gi)
                .sum();

            for (i, idx) in indices.into_iter().enumerate() {
                grad_input[idx] = s_slice[i] * (upstream_slice[i] - dot);
            }
        }

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
            .borrow()
            .iter()
            .zip(self.target.data.borrow().iter())
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
    assert!(
        tensor.shape == vec![],
        "[autodiff] Called backward on non-scalar tensor"
    );
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
                for (parent, parent_new_grad) in node.parents.iter().zip(parent_grads.iter()) {
                    if is_debug() {
                        println!(
                            "[autodiff]   - Parent: {:?} with grad {:?}",
                            parent, parent_new_grad
                        );
                    }
                    let mut parent_grad = parent.grad.borrow_mut();
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
                                println!("[autodiff]   - Setting new grad: {:?}", parent_new_grad);
                            }
                            parent_grad.replace(parent_new_grad.clone());
                        }
                    }
                }
            }
        }
    }
}
