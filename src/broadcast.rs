use crate::is_debug;
use crate::tensor::{Tensor, TensorRef};
use std::rc::Rc;

pub struct BroadcastIterator {
    left: Rc<Tensor>,
    right: Rc<Tensor>,
    out_shape: Vec<usize>,
    index: usize,
    total: usize,
}

// The idea of this operation is to translate the outputs flat index into
// the flat index for both the input matrices into the operation
// It also does broadcasting, but the main idea is the above
impl BroadcastIterator {
    pub fn new(left: &TensorRef, right: &TensorRef) -> Self {
        let out_shape = broadcast_shape(&left.shape, &right.shape);
        let total = out_shape.iter().product();

        if is_debug() {
            println!("Creating BroadcastIterator:");
            println!("  Left shape: {:?}", left.shape);
            println!("  Right shape: {:?}", right.shape);
            println!("  Output shape: {:?}", out_shape);
            println!("  Total elements: {}", total);
        }

        let left = left.0.clone();
        let right = right.0.clone();
        BroadcastIterator {
            left,
            right,
            out_shape,
            index: 0,
            total,
        }
    }
}

impl Iterator for BroadcastIterator {
    type Item = (usize, usize);

    fn next(&mut self) -> Option<Self::Item> {
        // Return None if we've processed all elements
        // println!(
        // "-------------------------- REQUESTING INDEX {} -------------",
        // self.index + 1
        // );
        if self.index >= self.total {
            return None;
        }

        // Calculate multi-dimensional indices from flat index
        let mut remaining = self.index;
        let mut left_index = 0;
        let mut right_index = 0;
        let mut stride_left = 1;
        let mut stride_right = 1;

        // Debug: core iteration parameters
        if is_debug() {
            println!("[DEBUG] Processing element {}/{}", self.index, self.total);
        }

        // Process dimensions from least significant to most significant
        for dim in (0..self.out_shape.len()).rev() {
            // Calculate index in this dimension
            let dim_size = self.out_shape[dim];
            let idx = remaining % dim_size;
            remaining /= dim_size;

            // println!("Index in dim {} is {}", dim, idx);

            if dim < self.left.shape.len() {
                if self.left.shape[dim] == dim_size {
                    // Normal case: use the calculated index
                    left_index += idx * stride_left;
                } else if self.left.shape[dim] == 1 {
                    // Broadcasting case: reuse the same element
                } else {
                    // This should never happen if broadcast_shape validation is correct
                    panic!("Unexpected shape mismatch in dimension {}", dim);
                }
            }

            // Calculate right index for this dimension
            if dim < self.right.shape.len() {
                if self.right.shape[dim] == dim_size {
                    // Normal case: use the calculated index
                    right_index += idx * stride_right;
                } else if self.right.shape[dim] == 1 {
                    // Broadcasting case: reuse the same element
                } else {
                    panic!("Unexpected shape mismatch in dimension {}", dim);
                }
            }

            // Update strides for next dimension
            if dim < self.left.shape.len() {
                stride_left *= self.left.shape[dim];
            }

            if dim < self.right.shape.len() {
                stride_right *= self.right.shape[dim];
            }

            if is_debug() {
                println!(
                    "[DEBUG] End of dim loop dim={} with\nlidx: {}, stl: {}\nridx: {}, str: {}",
                    dim, left_index, stride_left, right_index, stride_right
                );
            }
        }

        if is_debug() {
            println!(
                "[DEBUG] Mapped flat index {} to (L:{}, R:{})",
                self.index, left_index, right_index
            );
        }

        // Increment index for next iteration
        self.index += 1;

        Some((left_index, right_index))
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.total - self.index;

        if is_debug() {
            println!("[broadcasting] size_hint: remaining={}", remaining);
        }

        (remaining, Some(remaining))
    }
}

/// Computes the broadcasted shape between two tensor shapes
///
/// Following numpy broadcasting rules:
/// - Start from trailing dimensions
/// - Dimensions are compatible if they're equal or one of them is 1
pub fn broadcast_shape(left_shape: &[usize], right_shape: &[usize]) -> Vec<usize> {
    if is_debug() {
        println!(
            "[broadcast] Broadcasting shapes: {:?} and {:?}",
            left_shape, right_shape
        );
    }

    // Get the maximum rank between the tensors
    let max_rank = std::cmp::max(left_shape.len(), right_shape.len());
    let mut result = Vec::with_capacity(max_rank);

    // Iterate from the trailing dimension
    for i in 0..max_rank {
        let left_idx = left_shape.len().saturating_sub(i + 1);
        let right_idx = right_shape.len().saturating_sub(i + 1);

        // Get dimensions (defaulting to 1 for missing dimensions)
        let left_dim = if left_idx < left_shape.len() {
            left_shape[left_idx]
        } else {
            1
        };
        let right_dim = if right_idx < right_shape.len() {
            right_shape[right_idx]
        } else {
            1
        };

        // Apply broadcasting rule - dimensions must be equal or one of them must be 1
        if left_dim == right_dim || left_dim == 1 || right_dim == 1 {
            let broadcast_dim = std::cmp::max(left_dim, right_dim);
            result.push(broadcast_dim);

            if is_debug() {
                println!(
                    "  Dimension {}: {} vs {} -> broadcasted to {}",
                    max_rank - i - 1,
                    left_dim,
                    right_dim,
                    broadcast_dim
                );
            }
        } else {
            panic!(
                "Cannot broadcast shapes {:?} and {:?} - incompatible dimensions {} and {}",
                left_shape, right_shape, left_dim, right_dim
            );
        }
    }

    // Reverse the result to get correct ordering (we processed from trailing dimension)
    result.reverse();

    if is_debug() {
        println!("[broadcast] Resulting broadcast shape: {:?}", result);
    }

    result
}
