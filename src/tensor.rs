use crate::autodiff;
use crate::autodiff::GradFn;
use std::fmt;
use std::{
    cell::RefCell,
    ops::{Add, Deref},
    rc::{Rc, Weak},
};

#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>, // data is stored in row-major order
    pub shape: Vec<usize>,
    pub requires_grad: bool,
    pub grad: RefCell<Option<Vec<f32>>>, // gradient of the tensor
    pub grad_fn: Option<Rc<dyn GradFn>>,
    pub parents: Vec<Weak<Tensor>>,
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
                "Data size must match shape"
            );
        }
        TensorRef(Rc::new(Tensor {
            data,
            shape,
            requires_grad: true,
            grad: RefCell::new(None),
            grad_fn: None,
            parents: vec![],
        }))
    }

    // Full constructor
    pub fn new_with_options(
        data: Vec<f32>,
        shape: Vec<usize>,
        requires_grad: bool,
        grad_fn: Option<Rc<dyn GradFn>>,
        parents: Vec<Weak<Tensor>>,
    ) -> TensorRef {
        assert_eq!(
            data.len(),
            shape.iter().product(),
            "Data size must match shape"
        );

        TensorRef(Rc::new(Tensor {
            data,
            shape,
            requires_grad,
            grad: RefCell::new(None),
            grad_fn,
            parents,
        }))
    }

    pub fn transpose(&self) -> TensorRef {
        assert_eq!(self.shape.len(), 2, "Transpose only supports 2D tensors");
        let rows = self.shape[0];
        let cols = self.shape[1];

        let mut data = vec![0.0 as f32; self.data.len()];
        let shape = vec![cols, rows];

        for i in 0..rows {
            for j in 0..cols {
                data[j * rows + i] = self.data[i * cols + j];
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
}

impl PartialEq for Tensor {
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data && self.shape == other.shape
    }
}

impl fmt::Debug for Tensor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Tensor")
            .field("data", &self.data)
            .field("shape", &self.shape)
            .field("grad", &self.grad.borrow())
            .field("requires_grad", &self.requires_grad)
            .field("parents", &self.parents)
            .finish()
    }
}
