use crate::tensor::tensor_ref::TensorRef;
use crate::tensor::GradFn;
use std::fmt;
use std::{
    cell::RefCell,
    rc::{Rc, Weak},
};

#[derive(Clone)]
pub struct Tensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub requires_grad: bool,
    pub grad: RefCell<Option<Vec<f32>>>,
    pub grad_fn: Option<Rc<dyn GradFn>>,
    pub parents: Vec<Weak<Tensor>>,
}

impl Tensor {
    pub fn new(data: Vec<f32>, shape: Vec<usize>) -> TensorRef {
        // accept scalars of shape []
        if shape.is_empty() {
            assert_eq!(data.len(), 1, "Data size must be 1 for scalar");
        } else {
            assert!(
                shape.len() > 1,
                "A Shape of 1 dim is ambiguous (either a row or column vector)"
            );
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
        let mut transposed_data = vec![0.0; self.data.len()];
        let rows = self.shape[0];
        let cols = self.shape[1];

        for i in 0..rows {
            for j in 0..cols {
                transposed_data[j * rows + i] = self.data[i * cols + j];
            }
        }

        Tensor::new(transposed_data, vec![cols, rows])
    }
}

// implment partialeq for tensor
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
