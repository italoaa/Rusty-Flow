use crate::tensor::{GradFn, Tensor};
use std::{
    cell::RefCell,
    fmt,
    ops::Deref,
    rc::{Rc, Weak},
};

pub struct TensorRef(pub Rc<Tensor>);

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

    pub fn matmul(&self, other: &TensorRef) -> TensorRef {
        assert_eq!(
            self.shape.len(),
            2,
            "Matrix multiplication only supports 2D tensors"
        );
        assert_eq!(
            other.shape.len(),
            2,
            "Matrix multiplication only supports 2D tensors"
        );
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

        Tensor::new(result, vec![self.shape[0], other.shape[1]])
    }
}

impl TensorRef {
    // Add this method to perform topological sorting
    // Add this method to perform topological sorting
    fn topo_sort(&self) -> Vec<Rc<Tensor>> {
        // println!("[topo_sort] Starting topological sort at node: {:?}", self);
        let mut result = Vec::new();
        let mut visited = std::collections::HashSet::new();
        let mut temp_mark = std::collections::HashSet::new();

        // Start DFS from the output node
        self.visit_node(self.0.clone(), &mut visited, &mut temp_mark, &mut result);

        // The result is in reverse topological order (exactly what we want for backprop)
        result
    }

    fn visit_node(
        &self,
        node: Rc<Tensor>,
        visited: &mut std::collections::HashSet<*const Tensor>,
        temp_mark: &mut std::collections::HashSet<*const Tensor>,
        result: &mut Vec<Rc<Tensor>>,
    ) {
        // println!("[topo_sort] Visiting node: {:?}", node);
        let node_ptr = Rc::as_ptr(&node);

        // Skip if already visited
        if visited.contains(&node_ptr) {
            return;
        }

        // Check for cycles (should not happen in a well-formed computational graph)
        if temp_mark.contains(&node_ptr) {
            panic!("Cycle detected in computational graph!");
        }

        // Mark temporarily until all parents are visited so we can detect cycles
        temp_mark.insert(node_ptr);

        // Visit all parents
        // println!("[topo_sort] Visiting parents of node: {:?}", node);
        // println!("[topo_sort] Parents: {:?}", node.parents);
        for parent_weak in &node.parents {
            if let Some(parent) = parent_weak.upgrade() {
                self.visit_node(parent, visited, temp_mark, result);
            }
        }

        // Mark as visited
        temp_mark.remove(&node_ptr);
        visited.insert(node_ptr);

        // Add to result
        // println!("[topo_sort] Adding node to result: {:?}", node);
        result.push(node);
        // println!("[topo_sort] Result {:?}", result);
    }

    pub fn backward(&self) {
        println!("[backward] Starting backward pass on: {:?}", self);

        // 1. First, set the output gradient to 1.0 if not already set
        {
            let mut grad_ref = self.grad.borrow_mut();
            if grad_ref.is_none() {
                println!("[backward] Setting output grad to 1.0 for {:?}", self.shape);
                grad_ref.replace(vec![1.0]);
            } else {
                println!(
                    "[backward] Output grad already set: {:?}",
                    grad_ref.as_ref()
                );
            }
        }

        // 2. Perform topological sort
        let sorted_nodes = self.topo_sort();

        for node in sorted_nodes.iter().rev() {
            if let Some(grad_fn) = &node.grad_fn {
                println!("[backward] Backwarding through node: {:?}", node);
                let grad = node.grad.borrow();

                if let Some(grad) = grad.as_ref() {
                    println!("[backward] Processing node: {:?}", node);
                    let parents_grad = grad_fn.backward(grad);

                    drop(grad);

                    // Distribute gradients to parents
                    for (i, (parent_weak, parent_new_grads)) in
                        node.parents.iter().zip(parents_grad.iter()).enumerate()
                    {
                        if let Some(parent_rc) = parent_weak.upgrade() {
                            println!(
                                "[backward] Propagating to parent {}: {:?}",
                                i, parent_new_grads
                            );

                            // Update parent's gradients with accumulation
                            let mut parent_grad = parent_rc.grad.borrow_mut();
                            match &mut *parent_grad {
                                Some(existing_grad) => {
                                    // Accumulate gradients
                                    for (e, n) in existing_grad.iter_mut().zip(parent_new_grads) {
                                        *e += n;
                                    }
                                }
                                None => {
                                    // Set initial gradient
                                    parent_grad.replace(parent_new_grads.clone());
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

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
