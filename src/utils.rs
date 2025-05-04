use crate::tensor::Tensor;
use std::collections::HashSet;
use std::rc::Rc;

// Two tensors are “broadcastable” if the following rules hold:

//     Each tensor has at least one dimension.

//     When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.

// If two tensors x, y are “broadcastable”, the resulting tensor size is calculated as follows:

//     If the number of dimensions of x and y are not equal, prepend 1 to the dimensions of the tensor with fewer dimensions to make them equal length.

//     Then, for each dimension size, the resulting dimension size is the max of the sizes of x and y along that dimension.

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
