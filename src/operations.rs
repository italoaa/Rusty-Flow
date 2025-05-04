use crate::autodiff::GradFn;
use crate::autodiff::{
    AddBack, CrossEntropyBack, CrossEntropyLogitsBack, DivBack, MMBack, MSEBack, MulBack, ReLUBack,
    SoftmaxBack, SubBack, SumBack,
};
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
        assert_eq!(self.shape, other.shape, "Shape mismatch for addition");
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
            Some(Rc::new(AddBack) as Rc<dyn GradFn>)
        } else {
            None
        };

        Tensor::new_with_options(data, self.shape.clone(), requires_grad, grad_fn, parents)
    }
}

// ================ SUBSTRACT OPERATION ==================

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
            Some(Rc::new(SubBack) as Rc<dyn GradFn>)
        } else {
            None
        };

        let parents = vec![Rc::downgrade(&self.0), Rc::downgrade(&other.0)];

        Tensor::new_with_options(data, self.shape.clone(), requires_grad, grad_fn, parents)
    }
}

// ================ MULTIPLY OPERATION ==================

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
            Some(Rc::new(MulBack {
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

// ================ DIVIDE OPERATION ==================

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
            Some(Rc::new(DivBack {
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

// ================ SUM OPERATION =========================

impl TensorRef {
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
}

// ================ Mean OPERATION ==================

impl TensorRef {
    // little helper
    pub fn requires_grad(&self) -> bool {
        self.requires_grad
    }

    pub fn mean(&self) -> TensorRef {
        let sum: TensorRef = self.sum();
        // Divide by the number of elements
        let len: usize = self.shape.iter().product();
        // no broadcasting
        let div = Tensor::new(vec![len as f32], vec![]);
        &sum / &div
    }
}

// ================ MatMul OPERATION ==================

impl TensorRef {
    pub fn mm(&self, other: &TensorRef) -> TensorRef {
        assert!(self.shape.len() < 3, "3d mm is not implemented");
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

        let requires_grad = self.requires_grad || other.requires_grad;
        let grad_fn = if requires_grad {
            Some(Rc::new(MMBack {
                // expects Rc<tensor>
                left: self.0.clone(),
                right: other.0.clone(),
            }) as Rc<dyn GradFn>)
        } else {
            None
        };

        let parents = vec![Rc::downgrade(&self.0), Rc::downgrade(&other.0)];

        Tensor::new_with_options(
            result,
            vec![self.shape[0], other.shape[1]],
            requires_grad,
            grad_fn,
            parents,
        )
    }
}

// ================ Activations ==================

// ReLU
impl TensorRef {
    pub fn relu(&self) -> TensorRef {
        let data = self
            .data
            .iter()
            .map(|&x| if x > 0.0 { x } else { 0.0 })
            .collect::<Vec<_>>();

        let requires_grad = self.requires_grad;

        let grad_fn = if requires_grad {
            Some(Rc::new(ReLUBack {
                input: self.0.clone(),
            }) as Rc<dyn GradFn>)
        } else {
            None
        };

        let parents = vec![Rc::downgrade(&self.0)];

        Tensor::new_with_options(data, self.shape.clone(), requires_grad, grad_fn, parents)
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
            .iter()
            .zip(target.data.iter())
            .map(|(a, b)| (a - b).powi(2))
            .collect::<Vec<_>>();

        let requires_grad = self.requires_grad || target.requires_grad;

        let grad_fn = if requires_grad {
            Some(Rc::new(MSEBack {
                input: self.0.clone(),
                target: target.0.clone(),
            }) as Rc<dyn GradFn>)
        } else {
            None
        };

        let parents = vec![Rc::downgrade(&self.0), Rc::downgrade(&target.0)];

        Tensor::new_with_options(data, self.shape.clone(), requires_grad, grad_fn, parents)
    }

    // Calculates the cross entropy between two distributions
    pub fn cross_entropy(&self, target: &TensorRef) -> TensorRef {
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
            for row in target.data.chunks_exact(n_classes) {
                let ones_count = row.iter().filter(|&&x| x == 1.0).count();
                assert_eq!(ones_count, 1, "Target tensor is not one-hot encoded");
            }
            for row in self.data.chunks_exact(n_classes) {
                let sum: f32 = row.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-6,
                    "Input tensor row is not a probability distribution"
                );
            }
        } else if is_1d {
            // 1D: target shape [num_classes]
            let ones_count = target.data.iter().filter(|&&x| x == 1.0).count();
            assert_eq!(ones_count, 1, "1D target tensor is not one-hot encoded");

            let sum: f32 = self.data.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "1D input tensor is not a probability distribution"
            );
        }

        // Calculate cross-entropy: same for both cases
        let data = self
            .data
            .iter()
            .zip(target.data.iter())
            .map(|(a, b)| -b * (a + 1e-6).ln())
            .collect::<Vec<_>>();

        let requires_grad = self.requires_grad || target.requires_grad;
        let grad_fn = if requires_grad {
            Some(Rc::new(CrossEntropyBack {
                input: self.0.clone(),
                target: target.0.clone(),
            }) as Rc<dyn GradFn>)
        } else {
            None
        };

        let parents = vec![Rc::downgrade(&self.0), Rc::downgrade(&target.0)];

        Tensor::new_with_options(data, self.shape.clone(), requires_grad, grad_fn, parents)
    }

    pub fn cross_entropy_with_logits(&self, target: &TensorRef) -> TensorRef {
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
            for row in target.data.chunks_exact(n_classes) {
                let ones_count = row.iter().filter(|&&x| x == 1.0).count();
                assert_eq!(ones_count, 1, "Target tensor is not one-hot encoded");
            }
            for row in self.data.chunks_exact(n_classes) {
                let sum: f32 = row.iter().sum();
                assert!(
                    (sum - 1.0).abs() < 1e-6,
                    "Input tensor row is not a probability distribution"
                );
            }
        } else if is_1d {
            // 1D: target shape [num_classes]
            let ones_count = target.data.iter().filter(|&&x| x == 1.0).count();
            assert_eq!(ones_count, 1, "1D target tensor is not one-hot encoded");

            let sum: f32 = self.data.iter().sum();
            assert!(
                (sum - 1.0).abs() < 1e-6,
                "1D input tensor is not a probability distribution"
            );
        }

        // First, apply softmax to the input tensor
        let self_softmax = self.softmax();

        // Calculate cross-entropy: same for both cases
        let data = self_softmax
            .data
            .iter()
            .zip(target.data.iter())
            .map(|(a, b)| -b * (a + 1e-6).ln())
            .collect::<Vec<_>>();

        let requires_grad = self.requires_grad || target.requires_grad;
        let grad_fn = if requires_grad {
            Some(Rc::new(CrossEntropyLogitsBack {
                softmax: self_softmax.0.clone(),
                target: target.0.clone(),
            }) as Rc<dyn GradFn>)
        } else {
            None
        };

        let parents = vec![Rc::downgrade(&self.0), Rc::downgrade(&target.0)];

        Tensor::new_with_options(data, self.shape.clone(), requires_grad, grad_fn, parents)
    }

    // Softmax function
    pub fn softmax(&self) -> TensorRef {
        let max_val = self.data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let exp_data: Vec<f32> = self.data.iter().map(|&x| (x - max_val).exp()).collect();
        let sum_exp: f32 = exp_data.iter().sum();
        let softmax_data: Vec<f32> = exp_data.iter().map(|&x| x / sum_exp).collect();

        let requires_grad = self.requires_grad;

        let grad_fn = if requires_grad {
            Some(Rc::new(SoftmaxBack {
                output: Tensor::new(softmax_data.clone(), self.shape.clone())
                    .0
                    .clone(),
            }) as Rc<dyn GradFn>)
        } else {
            None
        };

        let parents = vec![Rc::downgrade(&self.0)];

        Tensor::new_with_options(
            softmax_data,
            self.shape.clone(),
            requires_grad,
            grad_fn,
            parents,
        )
    }
}

// ================== Loss Functions ==================
