use crate::autodiff;
use crate::autodiff::GradFn;
use crate::utils::{compute_strides, slices_along_dim};
use rand::prelude::*;
use rand_distr::StandardNormal;

use std::fmt;
use std::{
    cell::RefCell,
    ops::{Add, Deref},
    rc::{Rc, Weak},
};

#[derive(Clone)]
pub struct Tensor {
    pub data: RefCell<Vec<f32>>, // data is stored in row-major order
    pub shape: Vec<usize>,
    pub strides: Vec<usize>, // stride is not used yet
    pub requires_grad: bool,
    pub grad: RefCell<Option<Vec<f32>>>, // gradient of the tensor
    pub grad_fn: Option<Rc<dyn GradFn>>,
    pub parents: Vec<Rc<Tensor>>,
}

pub struct TensorRef(pub Rc<Tensor>);

// Basic traits
impl PartialEq for TensorRef {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl fmt::Debug for TensorRef {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0.fmt(f)
    }
}
impl Deref for TensorRef {
    type Target = Tensor;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

// Reference to tensor object
impl TensorRef {
    // modifies itself
    pub fn transpose(&self) -> TensorRef {
        self.0.transpose()
    }
    pub fn backward(&self) {
        autodiff::backward(self);
    }

    pub fn set_grad(&self, grads: Vec<f32>) {
        assert_eq!(
            grads.len(),
            self.data.borrow().len(),
            "Gradient size must match data size"
        );
        self.grad.replace(Option::Some(grads));
    }
    pub fn rc(&self) -> Rc<Tensor> {
        self.0.clone()
    }
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> TensorRef {
        // accept scalars of shape []
        if shape.is_empty() {
            assert_eq!(data.len(), 1, "Data size must be 1 for scalar");
        } else {
            assert_eq!(
                data.len(),
                shape.iter().product(),
                "Data size must match shape {:?}, {} != {}, (data: {:?})",
                shape,
                data.len(),
                shape.iter().product::<usize>(),
                data
            );
        }

        Tensor::new_with_options(
            data,
            shape,
            true, // by default requires_grad is true
            None,
            vec![],
        )
    }

    pub fn new_nograd(data: Vec<f32>, shape: Vec<usize>) -> TensorRef {
        // accept scalars of shape []
        if shape.is_empty() {
            assert_eq!(data.len(), 1, "Data size must be 1 for scalar");
        } else {
            assert_eq!(
                data.len(),
                shape.iter().product(),
                "Data size must match shape {:?}, {} != {}, (data: {:?})",
                shape,
                data.len(),
                shape.iter().product::<usize>(),
                data
            );
        }

        Tensor::new_with_options(
            data,
            shape,
            false, // by default requires_grad is false
            None,
            vec![],
        )
    }

    // Full constructor
    pub fn new_with_options(
        data: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: bool,
        grad_fn: Option<Rc<dyn GradFn>>,
        parents: Vec<Rc<Tensor>>,
    ) -> TensorRef {
        // data its len and the shape
        assert_eq!(
            data.len(),
            shape.iter().product(),
            "Data size must match shape"
        );
        let strides = compute_strides(&shape);

        TensorRef(Rc::new(Tensor {
            data: data.into(),
            shape,
            strides,
            requires_grad,
            grad: RefCell::new(None),
            grad_fn,
            parents,
        }))
    }

    // Mainly used for making biases so requires_grad is true
    pub fn ones_like(shape: Vec<usize>) -> TensorRef {
        let size = shape.iter().product();
        let data = vec![1.0; size];

        Tensor::new(data, shape)
    }

    pub fn zeros_like(shape: Vec<usize>) -> TensorRef {
        let size = shape.iter().product();
        let data = vec![0.0; size];

        Tensor::new(data, shape)
    }

    // Sample from a normal distribution
    // mainly used to make weights so requires_grad is true
    pub fn new_random(shape: Vec<usize>) -> TensorRef {
        let size = shape.iter().product();

        let data: Vec<f32> = (0..size)
            .map(|_| {
                let val: f32 = rand::rng().sample(StandardNormal);
                val
            })
            .collect();

        Tensor::new(data, shape)
    }

    pub fn xavier_init(shape: Vec<usize>) -> TensorRef {
        assert!(shape.len() >= 2, "Xavier init requires at least 2D shape");

        let fan_in = shape[shape.len() - 2];
        let fan_out = shape[shape.len() - 1];
        let limit = (6.0 / (fan_in + fan_out) as f32).sqrt();

        let size = shape.iter().product();
        let data: Vec<f32> = (0..size)
            .map(|_| rand::rng().gen_range(-limit..limit))
            .collect();

        Tensor::new(data, shape)
    }

    pub fn kaiming_init(shape: Vec<usize>, negative_slope: f32) -> TensorRef {
        assert!(shape.len() >= 2, "Kaiming init requires at least 2D shape");

        let fan_in = shape[shape.len() - 2];
        let std = (2.0 / ((1.0 + negative_slope.powi(2)) * fan_in as f32)).sqrt();

        let size = shape.iter().product();
        let data: Vec<f32> = (0..size)
            .map(|_| rand::rng().sample::<f32, _>(StandardNormal) * std)
            .collect();

        Tensor::new(data, shape)
    }

    pub fn transpose(&self) -> TensorRef {
        assert_eq!(self.shape.len(), 2, "Transpose only supports 2D tensors");
        let rows = self.shape[0];
        let cols = self.shape[1];

        let mut data = vec![0.0 as f32; self.data.borrow().len()];
        let shape = vec![cols, rows];

        for i in 0..rows {
            for j in 0..cols {
                data[j * rows + i] = self.data.borrow()[i * cols + j];
            }
        }

        Tensor::new_with_options(
            data,
            shape,
            self.requires_grad,
            // self.grad_fn.clone(),
            // grad fn for transpose is not implemented yet
            None,
            self.parents.clone(),
        )
    }

    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape
    }
}

impl Tensor {
    pub fn approx_eq(&self, other: &TensorRef, tol: f32) -> bool {
        if self.shape != other.shape || self.data.borrow().len() != other.data.borrow().len() {
            return false;
        }
        self.data
            .borrow()
            .iter()
            .zip(other.data.borrow().iter())
            .all(|(a, b)| (a - b).abs() <= tol)
    }

    pub fn clear_grad(&self) {
        self.grad.replace(None);
        for parent in &self.parents {
            parent.clear_grad();
        }
    }

    pub fn normalize() {}
}

impl Tensor {
    pub fn iterate_over_dim<'a>(&'a self, dim: usize) -> impl Iterator<Item = Vec<f32>> + 'a {
        let data = &self.data;
        slices_along_dim(&self.shape, dim)
            .map(move |indices| indices.into_iter().map(|i| data.borrow()[i]).collect())
    }

    pub fn iterate_over_dim_indices<'a>(
        &'a self,
        dim: usize,
    ) -> impl Iterator<Item = Vec<usize>> + 'a {
        slices_along_dim(&self.shape, dim)
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // Helper function to recursively format the tensor's data
        fn format_data(data: &[f32], shape: &[usize], indent: usize) -> String {
            match shape.len() {
                0 => format!("{:.4}", data[0]), // scalar: display as a single value
                1 => {
                    // 1D tensor: display as a row
                    let elements: Vec<String> = data.iter().map(|x| format!("{:.4}", x)).collect();
                    format!("[{}]", elements.join(", "))
                }
                _ => {
                    // nD tensor: recursively format sub-tensors
                    let chunk_size = shape[1..].iter().product();
                    let mut result = String::new();
                    result.push_str(&format!("[\n"));
                    for (i, chunk) in data.chunks(chunk_size).enumerate() {
                        result.push_str(&" ".repeat(indent + 2));
                        result.push_str(&format_data(chunk, &shape[1..], indent + 2));
                        if i < data.chunks(chunk_size).count() - 1 {
                            result.push_str(",\n");
                        }
                    }
                    result.push_str(&format!("\n{}]", " ".repeat(indent)));
                    result
                }
            }
        }

        // Format the tensor's data
        let data_str = format_data(&self.data.borrow(), &self.shape, 0);
        // Shorten the data string if it's too long
        let max_length = 100;
        let data_str = if data_str.len() > max_length {
            format!("{}...", &data_str[..max_length])
        } else {
            data_str
        };

        // Format the other fields
        let grad_str = if let Some(grad) = self.grad.borrow().as_ref() {
            format!("\n  grad: {:?}", grad)
        } else {
            String::new()
        };

        let parents_str = if !self.parents.is_empty() {
            // iterate through the parents and print their shapes
            let parents_shapes: Vec<String> = self
                .parents
                .iter()
                .map(|parent| format!("{:?}", parent.shape))
                .collect();
            format!("\n  parents: [{}]", parents_shapes.join(", "))
        } else {
            String::new()
        };

        // Combine all fields into a single output
        write!(
            f,
            "Tensor(\n  data: {},\n  shape: {:?},\n  requires_grad: {}{}{}\n)",
            data_str, self.shape, self.requires_grad, grad_str, parents_str
        )
    }
}
