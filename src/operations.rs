use crate::autodiff::GradFn;
use crate::autodiff::{
    AddBack, CrossEntropyBack, CrossEntropyLogitsBack, DivBack, MMBack, MSEBack, MeanBack, MulBack,
    ReLUBack, SoftmaxBack, SubBack, SumBack,
};
use crate::broadcast::{broadcast_shape, BroadcastIterator};
use crate::is_debug;
use crate::tensor::{Tensor, TensorRef};
use std::{
    cell::RefCell,
    collections::HashSet,
    fmt,
    ops::{Add, Deref, Div, Mul, Sub},
    rc::{Rc, Weak},
};

// ================ ADD OPERATION ==================

impl<'a, 'b> Add<&'b TensorRef> for &'a TensorRef {
    type Output = TensorRef;

    fn add(self, other: &'b TensorRef) -> TensorRef {
        if is_debug() {
            println!("[add] Adding tensors {:?} and {:?}", self, other);
        }

        let mut data: Vec<f32> = Vec::new();
        let mut iter = BroadcastIterator::new(&self, &other);
        while let Some((i, j)) = iter.next() {
            data.push(self.data.borrow()[i] + other.data.borrow()[j]);
        }

        let requires_grad = self.requires_grad || other.requires_grad;

        // logging parents
        let parents = vec![self.rc(), other.rc()];
        // println!("[add] New has parents: {:?}", parents);

        let grad_fn = if requires_grad {
            Some(Rc::new(AddBack {
                left_shape: self.shape.clone(),
                right_shape: other.shape.clone(),
                out_shape: broadcast_shape(&self.shape, &other.shape),
            }) as Rc<dyn GradFn>)
        } else {
            None
        };

        let output_shape = broadcast_shape(&self.shape, &other.shape);

        Tensor::new_with_options(data, output_shape, requires_grad, grad_fn, parents)
    }
}

// ================ SUBSTRACT OPERATION ==================

impl<'a, 'b> Sub<&'b TensorRef> for &'a TensorRef {
    type Output = TensorRef;

    fn sub(self, other: &'b TensorRef) -> TensorRef {
        if is_debug() {
            println!("[sub] Subtracting tensors {:?} and {:?}", self, other);
        }

        let mut data: Vec<f32> = Vec::new();
        let mut iter = BroadcastIterator::new(&self, &other);
        while let Some((i, j)) = iter.next() {
            data.push(self.data.borrow()[i] - other.data.borrow()[j]);
        }

        let requires_grad = self.requires_grad || other.requires_grad;

        let grad_fn = if requires_grad {
            Some(Rc::new(SubBack {
                left_shape: self.shape.clone(),
                right_shape: other.shape.clone(),
                out_shape: broadcast_shape(&self.shape, &other.shape),
            }) as Rc<dyn GradFn>)
        } else {
            None
        };

        let parents = vec![self.rc(), other.rc()];

        let output_shape = broadcast_shape(&self.shape, &other.shape);

        Tensor::new_with_options(data, output_shape, requires_grad, grad_fn, parents)
    }
}

// ================ MULTIPLY OPERATION ==================

impl<'a, 'b> Mul<&'b TensorRef> for &'a TensorRef {
    type Output = TensorRef;

    fn mul(self, other: &'b TensorRef) -> TensorRef {
        if is_debug() {
            println!("[mul] Multiplying tensors {:?} and {:?}", self, other);
        }

        let mut data: Vec<f32> = Vec::new();
        let mut iter = BroadcastIterator::new(&self, &other);
        while let Some((i, j)) = iter.next() {
            data.push(self.data.borrow()[i] * other.data.borrow()[j]);
        }

        let requires_grad = self.requires_grad || other.requires_grad;

        let grad_fn = if requires_grad {
            Some(Rc::new(MulBack {
                left: self.rc(),
                right: other.rc(),
                out_shape: broadcast_shape(&self.shape, &other.shape),
            }) as Rc<dyn GradFn>)
        } else {
            None
        };
        let parents = vec![self.rc(), other.rc()];

        let output_shape = broadcast_shape(&self.shape, &other.shape);

        Tensor::new_with_options(data, output_shape, requires_grad, grad_fn, parents)
    }
}

// ================ DIVIDE OPERATION ==================

impl<'a, 'b> Div<&'b TensorRef> for &'a TensorRef {
    type Output = TensorRef;

    fn div(self, other: &'b TensorRef) -> TensorRef {
        if is_debug() {
            println!("[div] Dividing tensors {:?} and {:?}", self, other);
        }

        let mut data: Vec<f32> = Vec::new();
        let mut iter = BroadcastIterator::new(&self, &other);
        while let Some((i, j)) = iter.next() {
            if other.data.borrow()[j] == 0.0 {
                panic!("Division by zero");
            }
            data.push(self.data.borrow()[i] / other.data.borrow()[j]);
        }

        let requires_grad = self.requires_grad || other.requires_grad;

        let grad_fn = if requires_grad {
            Some(Rc::new(DivBack {
                left: self.rc(),
                right: other.rc(),
                out_shape: broadcast_shape(&self.shape, &other.shape),
            }) as Rc<dyn GradFn>)
        } else {
            None
        };

        let parents = vec![self.rc(), other.rc()];

        let output_shape = broadcast_shape(&self.shape, &other.shape);

        Tensor::new_with_options(data, output_shape, requires_grad, grad_fn, parents)
    }
}

// ================ SUM OPERATION =========================

impl TensorRef {
    pub fn sum(&self, dim: usize) -> TensorRef {
        // Sum the tensor along the specified dimension.
        // Example: sum([1, 2, 3], dim=0) = 6
        // Example: sum([[1, 2], [3, 4]], dim=0) = [4, 6]
        // Example: sum([[1, 2], [3, 4]], dim=1) = [3, 7]
        assert!(
            dim < self.shape.len(),
            "Dimension out of range. Tensor has {} dimensions, but {} was given.",
            self.shape.len(),
            dim
        );

        let out_shape = self
            .shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != dim)
            .map(|(_, &s)| s)
            .collect::<Vec<_>>();

        let mut result = Vec::with_capacity(out_shape.iter().product());

        for slice in self.iterate_over_dim(dim) {
            let sum: f32 = slice.iter().sum();
            result.push(sum);
        }

        let requires_grad = self.requires_grad;

        let grad_fn = if requires_grad {
            Some(Rc::new(SumBack {
                input_shape: self.shape.clone(),
                dim: dim,
            }) as Rc<dyn GradFn>)
        } else {
            None
        };

        if is_debug() {
            // Initial
            println!("[sum] Initial tensor: {:?}", self);
            println!("[sum] Sum result: {:?}", result);
        }

        let parents = vec![self.rc()];

        Tensor::new_with_options(result.clone(), out_shape, true, grad_fn, parents)
    }
}

// ================ Mean OPERATION ==================

impl TensorRef {
    // little helper
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn mean(&self, dim: usize) -> TensorRef {
        assert!(
            dim < self.shape.len(),
            "Dimension out of range. Tensor has {} dimensions, but {} was given.",
            self.shape.len(),
            dim
        );

        let out_shape = self
            .shape
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != dim)
            .map(|(_, &s)| s)
            .collect::<Vec<_>>();

        let mut result = Vec::with_capacity(out_shape.iter().product());

        for slice in self.iterate_over_dim(dim) {
            let sum: f32 = slice.iter().sum();
            let len: usize = slice.len();
            result.push(sum / len as f32);
        }

        let requires_grad = self.requires_grad;

        let grad_fn = if requires_grad {
            Some(Rc::new(MeanBack {
                input_shape: self.shape.clone(),
                dim: dim,
            }) as Rc<dyn GradFn>)
        } else {
            None
        };

        let parents = vec![self.rc()];

        Tensor::new_with_options(result.clone(), out_shape, requires_grad, grad_fn, parents)
    }
}

// ================ MatMul OPERATION ==================

impl TensorRef {
    pub fn mm(&self, other: &TensorRef) -> TensorRef {
        assert!(
            self.shape.len() >= 2 && other.shape.len() >= 2,
            "Matrix multiplication requires at least 2D tensors"
        );

        let m = self.shape[self.shape.len() - 2];
        let k = self.shape[self.shape.len() - 1];
        let n = other.shape[other.shape.len() - 1];
        assert_eq!(
            k,
            other.shape[other.shape.len() - 2],
            "Incompatible shapes for matrix multiplication"
        );

        let left_batch = &self.shape[..self.shape.len() - 2];
        let right_batch = &other.shape[..other.shape.len() - 2];
        let out_batch = broadcast_shape(left_batch, right_batch);

        let mut output_shape = out_batch.clone();
        output_shape.push(m);
        output_shape.push(n);

        let mut result = Vec::with_capacity(output_shape.iter().product());

        let mut iter = BroadcastIterator::new_with_shapes(left_batch, right_batch);

        // Iterate over the batches
        while let Some((i, j)) = iter.next() {
            let left_batch_offset = i * m * k;
            let right_batch_offset = j * k * n;

            // Perform matrix multiplication for this batch
            for row in 0..m {
                for col in 0..n {
                    let mut sum = 0.0;
                    for idx in 0..k {
                        let left_idx = left_batch_offset + row * k + idx;
                        let right_idx = right_batch_offset + idx * n + col;
                        sum += self.data.borrow()[left_idx] * other.data.borrow()[right_idx];
                    }
                    result.push(sum);
                }
            }
        }

        let requires_grad = self.requires_grad || other.requires_grad;
        let grad_fn = if requires_grad {
            Some(Rc::new(MMBack {
                // expects Rc<tensor>
                left: self.rc(),
                right: other.rc(),
                out_shape: output_shape.clone(),
            }) as Rc<dyn GradFn>)
        } else {
            None
        };

        let parents = vec![self.rc(), other.rc()];

        Tensor::new_with_options(result, output_shape, requires_grad, grad_fn, parents)
    }
}

// ================ Activations ==================

// ReLU
impl TensorRef {
    pub fn relu(&self) -> TensorRef {
        let data = self
            .data
            .borrow()
            .iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect::<Vec<_>>();

        let requires_grad = self.requires_grad;

        let grad_fn = if requires_grad {
            Some(Rc::new(ReLUBack { input: self.rc() }) as Rc<dyn GradFn>)
        } else {
            None
        };

        let parents = vec![self.rc()];

        Tensor::new_with_options(data, self.shape.clone(), requires_grad, grad_fn, parents)
    }
}

// ==================== Encodings ==================

impl TensorRef {
    pub fn one_hot(&self, num_classes: usize) -> TensorRef {
        assert!(
            self.shape.len() == 2,
            "One-hot encoding only supports 2D tensors"
        );

        assert_eq!(
            self.shape[1], 1,
            "One-hot encoding only supports tensors with shape [batch_size, 1]"
        );

        assert!(
            self.requires_grad == false,
            "One-hot encoding does not support tensors with requires_grad, maybe yet?"
        );

        let mut data = vec![0.0; self.shape[0] * num_classes];

        for (i, &value) in self.data.borrow().iter().enumerate() {
            let index = value as usize;
            assert!(
                index < num_classes,
                "Value {} is out of range for one-hot encoding with {} classes",
                value,
                num_classes
            );
            data[i * num_classes + index] = 1.0;
        }

        Tensor::new_with_options(
            data,
            vec![self.shape[0], num_classes],
            self.requires_grad,
            None,
            vec![self.rc()],
        )
    }
}

// ================ Comparison Operations ==================
// in here there is a lot of naming convention where if the function name includes loss then the
// output of the operation will be a scalar
// for example there is a mse and mse_loss the mse will return a tensor of the same shape as the input
// and the mse_loss will return a scalar

impl TensorRef {
    pub fn mse(&self, target: &TensorRef) -> TensorRef {
        assert_eq!(self.shape, target.shape);
        let data = self
            .data
            .borrow()
            .iter()
            .zip(target.data.borrow().iter())
            .map(|(a, b)| (a - b).powi(2))
            .collect::<Vec<_>>();

        let requires_grad = self.requires_grad || target.requires_grad;

        let grad_fn = if requires_grad {
            Some(Rc::new(MSEBack {
                input: self.rc(),
                target: target.rc(),
            }) as Rc<dyn GradFn>)
        } else {
            None
        };

        let parents = vec![self.rc(), target.rc()];

        Tensor::new_with_options(data, self.shape.clone(), requires_grad, grad_fn, parents)
    }

    // Calculates the cross entropy between two distributions
    pub fn cross_entropy(&self, target: &TensorRef) -> TensorRef {
        // Only accepts 2d shapes
        // the first is batch and second is the
        // probability of each class

        // NOTE: it only accepts one-hot encoded vecs

        // Check shapes: must match exactly
        assert_eq!(self.shape, target.shape);

        let is_2d = self.shape.len() == 2;
        let is_1d = self.shape.len() == 1;

        assert!(
            is_1d || is_2d,
            "Only 1D or 2D tensors are supported by cross_entropy"
        );
        assert_eq!(self.shape, target.shape);

        if is_2d {
            let n_classes = self.shape[1];
            for row in target.data.borrow().chunks_exact(n_classes) {
                let ones_count = row.iter().filter(|&&x| x == 1.0).count();
                assert_eq!(ones_count, 1, "Target tensor is not one-hot encoded");
            }
            for row in self.data.borrow().chunks_exact(n_classes) {
                let sum: f32 = row.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-6,
                    "Input tensor row is not a probability distribution, from row: {:?}, summed: {:?}",
                    row, sum
                );
            }
        } else if is_1d {
            // 1D: target shape [num_classes]
            let ones_count = target.data.borrow().iter().filter(|&&x| x == 1.0).count();
            assert_eq!(ones_count, 1, "1D target tensor is not one-hot encoded");

            let sum: f32 = self.data.borrow().iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "1D input tensor is not a probability distribution"
            );
        }

        // Calculate cross-entropy: same for both cases
        let data = self
            .data
            .borrow()
            .iter()
            .zip(target.data.borrow().iter())
            .map(|(a, b)| -b * (a + 1e-6).ln())
            .collect::<Vec<_>>();

        let requires_grad = self.requires_grad || target.requires_grad;
        let grad_fn = if requires_grad {
            Some(Rc::new(CrossEntropyBack {
                input: self.rc(),
                target: target.rc(),
            }) as Rc<dyn GradFn>)
        } else {
            None
        };

        let parents = vec![self.rc(), target.rc()];

        Tensor::new_with_options(data, self.shape.clone(), requires_grad, grad_fn, parents)
    }

    pub fn cross_entropy_with_logits(&self, target: &TensorRef) -> TensorRef {
        // Check shapes: must match exactly
        assert_eq!(
            self.shape, target.shape,
            "Input and target tensors must have the same shape but found shapes {:?} and {:?}",
            self.shape, target.shape
        );

        let is_2d = self.shape.len() == 2;
        let is_1d = self.shape.len() == 1;

        assert!(
            is_1d || is_2d,
            "Only 1D or 2D tensors are supported by cross_entropy"
        );
        assert_eq!(self.shape, target.shape);

        if is_2d {
            let n_classes = self.shape[1];
            for row in target.data.borrow().chunks_exact(n_classes) {
                let ones_count = row.iter().filter(|&&x| x == 1.0).count();
                assert_eq!(ones_count, 1, "Target tensor is not one-hot encoded");
            }
        } else if is_1d {
            // 1D: target shape [num_classes]
            let ones_count = target.data.borrow().iter().filter(|&&x| x == 1.0).count();
            assert_eq!(ones_count, 1, "1D target tensor is not one-hot encoded");
        }

        // First, apply softmax to the input tensor
        let last_dim = self.shape.len() - 1;
        let num_classes = self.shape[last_dim];
        let self_softmax = self.softmax(last_dim);

        // Calculate cross-entropy: same for both cases
        let data = self_softmax
            .data
            .borrow()
            .iter()
            .zip(target.data.borrow().iter())
            .map(|(a, b)| -b * (a + 1e-6).ln())
            .collect::<Vec<_>>();

        // Aggregate the value for the batch
        let data = data
            .chunks(num_classes)
            .map(|chunk| chunk.iter().sum())
            .collect::<Vec<_>>();

        let requires_grad = self.requires_grad || target.requires_grad;
        let grad_fn = if requires_grad {
            Some(Rc::new(CrossEntropyLogitsBack {
                softmax: self_softmax.rc(),
                target: target.rc(),
            }) as Rc<dyn GradFn>)
        } else {
            None
        };

        let parents = vec![self.rc(), target.rc()];

        Tensor::new_with_options(data, vec![self.shape[0]], requires_grad, grad_fn, parents)
    }

    // on a dim
    pub fn softmax(&self, dim: usize) -> TensorRef {
        let mut output_data = vec![0.0; self.data.borrow().len()];

        for indices in self.iterate_over_dim_indices(dim) {
            let group: Vec<f32> = indices.iter().map(|&i| self.data.borrow()[i]).collect();

            let max_val = group.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = group.iter().map(|&x| (x - max_val).exp()).collect();
            let sum_exps: f32 = exps.iter().sum();
            let softmax_group: Vec<f32> = exps.into_iter().map(|x| x / sum_exps).collect();

            for (&index, &value) in indices.iter().zip(softmax_group.iter()) {
                output_data[index] = value;
            }
        }

        let requires_grad = self.requires_grad;

        let grad_fn = if requires_grad {
            Some(Rc::new(SoftmaxBack {
                output: Tensor::new(output_data.clone(), self.shape.clone())
                    .0
                    .clone(),
                dim,
            }) as Rc<dyn GradFn>)
        } else {
            None
        };

        let parents = vec![self.rc()];

        Tensor::new_with_options(
            output_data,
            self.shape.clone(),
            requires_grad,
            grad_fn,
            parents,
        )
    }
}

// ================== Loss Functions ==================
