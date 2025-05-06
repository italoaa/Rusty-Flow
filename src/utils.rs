use crate::tensor::Tensor;
use std::collections::HashSet;
use std::rc::Rc;

pub fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![0; shape.len()];
    let mut stride = 1;

    for i in (0..shape.len()).rev() {
        strides[i] = stride;
        stride *= shape[i];
    }

    strides
}

// returns flat indexes of that slice as a vector
pub fn slices_along_dim(shape: &[usize], dim: usize) -> impl Iterator<Item = Vec<usize>> + use<'_> {
    let strides = compute_strides(&shape);

    // Calculate the number of iterations needed for each dimension except `dim`
    let num_iterations: usize = shape
        .iter()
        .enumerate()
        .filter(|&(i, _)| i != dim)
        .map(|(_, &size)| size)
        .product();

    // Generate the indices for each iteration
    (0..num_iterations).map(move |i| {
        let mut index = 0;
        let mut remaining = i;
        for (j, &stride) in strides.iter().enumerate() {
            if j != dim {
                let size = shape[j];
                let pos = remaining % size;
                index += pos * stride;
                remaining /= size;
            }
        }

        // Generate indices along the specified dimension
        (0..shape[dim])
            .map({
                let value = strides.clone();
                move |k| index + k * value[dim]
            })
            .collect()
    })
}

pub fn topo_sort(root: &Rc<Tensor>) -> Vec<Rc<Tensor>> {
    let mut result = Vec::new();
    let mut visited = HashSet::new();
    let mut temp = HashSet::new();
    visit_node(root.clone(), &mut visited, &mut temp, &mut result);
    result
}

fn visit_node(
    node: Rc<Tensor>,
    visited: &mut HashSet<*const Tensor>,
    temp: &mut HashSet<*const Tensor>,
    result: &mut Vec<Rc<Tensor>>,
) {
    let node_ptr = Rc::as_ptr(&node);
    if visited.contains(&node_ptr) {
        return;
    }

    if temp.contains(&node_ptr) {
        panic!("[utils] Cycle detected in computational graph");
    }
    temp.insert(node_ptr);
    for parent_weak in &node.parents {
        if let Some(parent) = parent_weak.upgrade() {
            visit_node(parent, visited, temp, result);
        }
    }
    temp.remove(&node_ptr);
    visited.insert(node_ptr);
    result.push(node);
}
