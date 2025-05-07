use crate::tensor::{Tensor, TensorRef};
use std::rc::Rc;

// optimizer struc

#[derive(Clone, Debug)]
pub struct SGD {
    pub parameters: Vec<Rc<Tensor>>,
    pub learning_rate: f32,
    pub momentum: f32,
    pub weight_decay: f32,
    pub velocities: Vec<Vec<f32>>, // Velocities for momentum
    dampening: f32,
}

impl SGD {
    pub fn new(
        parameters: Vec<Rc<Tensor>>,
        learning_rate: f32,
        momentum: f32,
        weight_decay: f32,
    ) -> Self {
        let velocities = parameters
            .iter()
            .map(|param| vec![0.0; param.data.borrow().len()])
            .collect();

        SGD {
            parameters,
            learning_rate,
            momentum,
            weight_decay,
            velocities,
            dampening: 0.0,
        }
    }
    pub fn new_with_options(
        parameters: Vec<Rc<Tensor>>,
        learning_rate: f32,
        momentum: f32,
        weight_decay: f32,
        dampening: f32,
    ) -> Self {
        let velocities = parameters
            .iter()
            .map(|param| vec![0.0; param.data.borrow().len()])
            .collect();

        SGD {
            parameters,
            learning_rate,
            momentum,
            weight_decay,
            velocities,
            dampening,
        }
    }

    pub fn step(&mut self) {
        for (param, velocity) in self.parameters.iter().zip(self.velocities.iter_mut()) {
            if let Some(grad) = param.grad.borrow().as_ref() {
                let mut data = param.data.borrow_mut();

                // Apply weight decay: g = g + weight_decay * d
                let grad_with_decay: Vec<f32> = grad
                    .iter()
                    .zip(data.iter())
                    .map(|(&g, &d)| g + self.weight_decay * d)
                    .collect();

                if self.momentum != 0.0 {
                    // if all the velocities are zero, initialize them
                    if velocity.iter().all(|&v| v == 0.0) {
                        for (v, g) in velocity.iter_mut().zip(grad_with_decay.iter()) {
                            *v = *g;
                        }
                    } else {
                        // Update velocity: v = momentum * v + (1 - dampening) * g
                        for (v, g) in velocity.iter_mut().zip(grad_with_decay.iter()) {
                            *v = *v * self.momentum + (1.0 - self.dampening) * g;
                        }
                    }
                }

                // Use momentum-adjusted gradient if applicable
                let final_grad = if self.momentum != 0.0 {
                    velocity.clone()
                } else {
                    grad_with_decay.clone()
                };

                // Update parameter: d = d - lr * g
                for (d, g) in data.iter_mut().zip(final_grad.iter()) {
                    *d -= self.learning_rate * g;
                }
            }
        }
    }
}
