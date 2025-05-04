use crate::is_debug;
use crate::tensor::{Tensor, TensorRef};
use std::rc::Rc;

pub struct BroadcastIterator {
    left_shape: Vec<usize>,
    right_shape: Vec<usize>,
    out_shape: Vec<usize>,
    index: usize,
    total: usize,
}

/// Resolves the index for the broadcasted tensor.
/// e.g. for a tensor of shape [2, 3, 4] and index 5,
/// the resolved index would be [0, 1, 1] (0*3*4 + 1*4 + 1)
/// This is done by iterating over the dimensions in reverse order.
pub fn resolve_broadcast_index(out_idx: usize, out_shape: &[usize], in_shape: &[usize]) -> usize {
    let mut remaining = out_idx;
    let mut idx = 0;
    let mut stride = 1;

    for dim in (0..out_shape.len()).rev() {
        let dim_size = out_shape[dim];
        let pos = remaining % dim_size;
        remaining /= dim_size;

        let in_dim = *in_shape.get(dim).unwrap_or(&1);
        let input_pos = if in_dim == 1 { 0 } else { pos };
        idx += input_pos * stride;

        if dim < in_shape.len() {
            stride *= in_shape[dim];
        }
    }

    idx
}

impl BroadcastIterator {
    pub fn new(left: &TensorRef, right: &TensorRef) -> Self {
        //iin shapes
        let out_shape = broadcast_shape(&left.shape, &right.shape);
        if is_debug() {
            println!("BroadcastIterator::new left: {:?}", left.shape);
            println!("BroadcastIterator::new right: {:?}", right.shape);
            println!("BroadcastIterator::new out_shape: {:?}", out_shape);
        }
        let total = out_shape.iter().product();
        BroadcastIterator {
            left_shape: left.shape.clone(),
            right_shape: right.shape.clone(),
            out_shape,
            index: 0,
            total,
        }
    }

    pub fn new_with_shapes(left_shape: &[usize], right_shape: &[usize]) -> Self {
        let out_shape = broadcast_shape(left_shape, right_shape);
        let total = out_shape.iter().product();
        BroadcastIterator {
            left_shape: left_shape.to_vec(),
            right_shape: right_shape.to_vec(),
            out_shape,
            index: 0,
            total,
        }
    }
}

impl Iterator for BroadcastIterator {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        if self.index >= self.total {
            return None;
        }

        let left_idx = resolve_broadcast_index(self.index, &self.out_shape, &self.left_shape);
        let right_idx = resolve_broadcast_index(self.index, &self.out_shape, &self.right_shape);

        self.index += 1;
        Some((left_idx, right_idx))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total - self.index;
        (remaining, Some(remaining))
    }
}

pub fn broadcast_shape(left: &[usize], right: &[usize]) -> Vec<usize> {
    let mut shape = Vec::new();
    let max_rank = left.len().max(right.len());

    for i in 0..max_rank {
        let l = *left.get(left.len().wrapping_sub(i + 1)).unwrap_or(&1);
        let r = *right.get(right.len().wrapping_sub(i + 1)).unwrap_or(&1);

        if l == r || l == 1 || r == 1 {
            shape.push(l.max(r));
        } else {
            panic!(
                "Cannot broadcast shapes {:?} and {:?}: incompatible dimensions {} and {}",
                left, right, l, r
            );
        }
    }

    shape.reverse();
    shape
}
