use std::{
    cell::RefCell,
    collections::VecDeque,
    f32, iter,
    mem::swap,
    ops::{Deref, DerefMut},
};

use rand::prelude::*;

pub struct Model {
    layers: Vec<Layer>,
    input_size: usize,
    output_size: usize,
}

impl Model {
    pub fn new(dimensions: &[usize]) -> Model {
        assert!(dimensions.len() > 2);
        assert!(!dimensions.contains(&0));

        let mut layers = Vec::with_capacity(dimensions.len() - 1);
        for i in 1..dimensions.len() {
            layers.push(Layer::new(dimensions[i - 1], dimensions[i]));
        }

        Model {
            layers,
            input_size: *dimensions.first().unwrap(),
            output_size: *dimensions.last().unwrap(),
        }
    }

    pub fn variables(&self) -> impl Iterator<Item = &f32> {
        self.layers.iter().flat_map(Layer::variables)
    }

    pub fn variables_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.layers.iter_mut().flat_map(Layer::variables_mut)
    }

    pub fn variable_count(&self) -> usize {
        self.layers.iter().map(Layer::variable_count).sum()
    }

    pub fn driver(&self) -> ModelDriver<&Model> {
        ModelDriver {
            model: self,
            outputs: vec![vec![]; self.layers.len()],
            scratch: RefCell::new(VecDeque::new()),
        }
    }

    pub fn driver_mut(&mut self) -> ModelDriver<&mut Model> {
        ModelDriver {
            outputs: vec![vec![]; self.layers.len()],
            scratch: RefCell::new(VecDeque::new()),
            model: self,
        }
    }
}

#[derive(Debug)]
pub struct Layer {
    weights: Vec<f32>,
    biases: Vec<f32>,
    input_size: usize,
    output_size: usize,
}

impl Layer {
    pub fn new(input_size: usize, output_size: usize) -> Layer {
        let mut source = rand::thread_rng();
        let mut weights = Vec::with_capacity(output_size * input_size);

        for _ in 0..output_size {
            for _ in 0..input_size {
                weights.push((source.gen::<u64>() % 100) as f32 / 1000.0)
            }
        }

        let mut biases = Vec::with_capacity(output_size);
        for _ in 0..output_size {
            biases.push((source.gen::<u64>() % 100) as f32 / 100000.0)
        }

        Layer {
            weights,
            biases,
            input_size,
            output_size,
        }
    }

    pub fn run(&self, input: &[f32], output: &mut [f32]) {
        assert!(input.len() == self.input_size);
        assert!(output.len() == self.output_size);

        for (i, activation) in output.iter_mut().enumerate() {
            // the dot product of weights and inputs
            let net_input = input
                .iter()
                .enumerate()
                .map(|(j, input)| self.weights[i * self.input_size + j] * input)
                .sum::<f32>();
            let biased_net_input = net_input + self.biases[i];
            *activation = biased_net_input.max(0.0);
        }
    }

    pub fn variables(&self) -> impl Iterator<Item = &f32> {
        self.weights.iter().chain(self.biases.iter())
    }

    pub fn variables_mut(&mut self) -> impl Iterator<Item = &mut f32> {
        self.weights.iter_mut().chain(self.biases.iter_mut())
    }

    pub fn variable_count(&self) -> usize {
        self.output_size * self.input_size + self.output_size
    }
}

pub struct ModelDriver<M> {
    model: M,
    outputs: Vec<Vec<f32>>,
    scratch: RefCell<VecDeque<f32>>,
}

impl<M> ModelDriver<M>
where
    M: Deref<Target = Model>,
{
    /// Gets a reference to the underlying model.
    pub fn model(&self) -> &Model {
        &self.model
    }

    /// Gets a mutable reference to the underlying model.
    pub fn model_mut(&mut self) -> &mut Model
    where
        M: DerefMut<Target = Model>,
    {
        &mut self.model
    }

    /// Runs the network on the given input.
    pub fn run<'a>(&'a mut self, input: &'a [f32]) -> RunResult<'a> {
        assert!(input.len() == self.model.input_size);

        let max_output_size = self
            .model
            .layers
            .iter()
            .map(|l| l.output_size)
            .max()
            .unwrap();

        let scratch = self.scratch.get_mut();
        scratch.clear();
        scratch.extend(iter::repeat(0.0).take(max_output_size * 2));
        let (mut a, mut b) = scratch.make_contiguous().split_at_mut(max_output_size);

        let first_layer = self.model.layers.first().unwrap();
        first_layer.run(input, &mut a[..first_layer.output_size]);

        for layer in self.model.layers.iter().skip(1) {
            layer.run(&a[..layer.input_size], &mut b[..layer.output_size]);
            swap(&mut a, &mut b);
        }

        RunResult { output: &a[..self.model.output_size] }
    }

    /// Runs the network on the given input and records the propagation through the network
    /// to allow for computation of the gradient. This is useful when training the network.
    pub fn run_and_record<'a>(&'a mut self, input: &'a [f32]) -> RecordedRunResult<'a> {
        assert!(input.len() == self.model.input_size);

        let mut layer_input = input;
        for (layer, output) in self.model.layers.iter().zip(self.outputs.iter_mut()) {
            output.clear();
            output.extend(iter::repeat(0.0).take(layer.output_size));

            layer.run(layer_input, output);
            layer_input = output;
        }

        RecordedRunResult {
            model: &self.model,
            input,
            outputs: &self.outputs,
            scratch: &self.scratch,
        }
    }

    /// A simple supervised learning training loop that accepts an iterator of `cases` containing
    /// input/output pairs, a configurable optimizer, and a configurable minibatch size.
    pub fn train<'a>(
        &mut self,
        cases: impl IntoIterator<Item = &'a (impl AsRef<[f32]> + 'a, impl AsRef<[f32]> + 'a)>,
        optimizer: impl Optimizer,
        batch_size: usize,
    ) where
        M: DerefMut<Target = Model>,
    {
        let variable_count = self.model.variable_count();

        let mut optimizer = optimizer.instance(variable_count);
        let mut dx = vec![0.0; variable_count];
        let mut t = 0;
        for (input, target) in cases {
            let (input, target): (&[f32], &[f32]) = (input.as_ref(), target.as_ref());

            t += 1;

            // feed the network the input and then add the gradients to `dx`
            self.run_and_record(input)
                .compute_gradients(target, |idx, val| dx[idx] += val);

            // only optimize after we have summed the gradients of `batch_size` cases
            if t % batch_size == 0 {
                // average the gradient
                dx.iter_mut().for_each(|dx| *dx /= batch_size as f32);
                // optimize the network
                optimizer.apply(self.model.variables_mut().zip(dx.iter()));
                // reset weight/bias grads for next batch
                dx.iter_mut().for_each(|dx| *dx = 0.0);
            }
        }

        // optimize again if there are remaining cases to be optimized with
        if t % batch_size != 0 {
            // average the gradient
            dx.iter_mut().for_each(|dx| *dx /= (t % batch_size) as f32);
            // optimize the network
            optimizer.apply(self.model.variables_mut().zip(dx.iter()));
        }
    }
}

/// The result from running the neural network on an input.
pub struct RunResult<'a> {
    output: &'a [f32],
}

impl<'a> RunResult<'a> {
    /// The network's final output for the given input.
    pub fn output(&self) -> &[f32] {
        self.output
    }
}

/// The result from running the neural network on an input and recording the propagation.
pub struct RecordedRunResult<'a> {
    model: &'a Model,
    input: &'a [f32],
    outputs: &'a Vec<Vec<f32>>,
    scratch: &'a RefCell<VecDeque<f32>>,
}

impl<'a> RecordedRunResult<'a> {
    /// The network's final output for the given input.
    pub fn output(&self) -> &[f32] {
        self.outputs.last().unwrap()
    }

    /// Backpropagates `target` into the network using the recorded propagation values
    /// to compute the gradient of all weights and biases of the network.
    pub fn compute_gradients(&self, target: &[f32], mut gradient: impl FnMut(usize, f32)) {
        let mut variable_count = self.model.variable_count();

        let mut node_grads = self.scratch.borrow_mut();

        node_grads.clear();
        node_grads.reserve(
            self.model
                .layers
                .iter()
                .map(|l| l.output_size)
                .max()
                .unwrap()
                * 2,
        );
        for (i, target) in target.iter().enumerate() {
            // derivative of 1/2 of squared error
            let dsqe = self.outputs.last().unwrap()[i] - target;
            node_grads.push_back(dsqe);
        }

        for (l, layer) in self.model.layers.iter().enumerate().rev() {
            assert!(node_grads.len() == layer.output_size);

            let prev_output = if l != 0 {
                self.outputs.get(l - 1)
            } else {
                None
            };

            // calculate gradient
            variable_count -= layer.variable_count();
            for i in 0..layer.output_size {
                for j in 0..layer.input_size {
                    let activation = prev_output.map(|o| o[j]).unwrap_or_else(|| self.input[j]);

                    gradient(
                        variable_count + i * layer.input_size + j,
                        node_grads[i] * activation,
                    );
                }

                gradient(
                    variable_count + layer.output_size * layer.input_size + i,
                    node_grads[i],
                );
            }

            // calculate prev layer's error (it's the next one we'll be visiting)
            if let Some(prev_output) = prev_output {
                let begin_grad_count = node_grads.len();

                for (j, prev_output) in prev_output.iter().copied().enumerate() {
                    let total_err = if prev_output > 0.0 {
                        node_grads
                            .iter()
                            .take(begin_grad_count)
                            .enumerate()
                            .map(|(i, delta)| layer.weights[i * layer.input_size + j] * delta)
                            .sum::<f32>()
                    } else {
                        0.0
                    };

                    node_grads.push_back(total_err / begin_grad_count as f32)
                }

                for _ in 0..begin_grad_count {
                    node_grads.pop_front();
                }
            }
        }
    }
}

pub trait Optimizer {
    type OptimizerInstance: OptimizerInstance;

    fn instance(self, n_vars: usize) -> Self::OptimizerInstance;
}

pub trait OptimizerInstance {
    fn apply<'a>(&mut self, vars_and_grads: impl Iterator<Item = (&'a mut f32, &'a f32)>);
}

pub trait StatelessOptimizer: OptimizerInstance {}

impl<T: StatelessOptimizer> Optimizer for T {
    type OptimizerInstance = Self;

    fn instance(self, _n_vars: usize) -> Self {
        self
    }
}

pub struct GradientDescentOptimizer {
    a: f32,
}

impl GradientDescentOptimizer {
    pub fn new(learn_rate: f32) -> GradientDescentOptimizer {
        GradientDescentOptimizer { a: learn_rate }
    }
}

impl StatelessOptimizer for GradientDescentOptimizer {}

impl OptimizerInstance for GradientDescentOptimizer {
    fn apply<'a>(&mut self, vars_and_grads: impl Iterator<Item = (&'a mut f32, &'a f32)>) {
        for (x, dx) in vars_and_grads {
            *x -= self.a * dx
        }
    }
}

pub struct AdamOptimizer {
    pub a: f32,
    pub b1: f32,
    pub b2: f32,
}

impl Optimizer for AdamOptimizer {
    type OptimizerInstance = AdamOptimizerInstance;

    fn instance(self, n_vars: usize) -> Self::OptimizerInstance {
        AdamOptimizerInstance {
            cfg: self,
            t: 0,
            vdx: vec![0.0; n_vars],
            sdx: vec![0.0; n_vars],
        }
    }
}

pub struct AdamOptimizerInstance {
    cfg: AdamOptimizer,
    t: i32,
    vdx: Vec<f32>,
    sdx: Vec<f32>,
}

impl OptimizerInstance for AdamOptimizerInstance {
    fn apply<'a>(&mut self, vars_and_grads: impl Iterator<Item = (&'a mut f32, &'a f32)>) {
        self.t += 1;

        for (((x, dx), vdx), sdx) in vars_and_grads
            .zip(self.vdx.iter_mut())
            .zip(self.sdx.iter_mut())
        {
            *vdx = self.cfg.b1 * *vdx + (1.0 - self.cfg.b1) * dx;
            *sdx = self.cfg.b2 * *sdx + (1.0 - self.cfg.b2) * dx * dx;

            let vdx_corr = *vdx / (1.0 - self.cfg.b1.powi(self.t));
            let sdx_corr = *sdx / (1.0 - self.cfg.b2.powi(self.t));

            *x -= self.cfg.a * vdx_corr / (sdx_corr.sqrt() + 1e-8);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn approx_sin() {
        let mut x = 0.0f32;
        let mut cases = Vec::new();
        while x < 1.0 {
            cases.push(([x], [(x * f32::consts::TAU).sin() + 1.0]));
            x += 0.0001;
        }

        let mut model = Model::new(&[1, 128, 1]);
        let mut driver = model.driver_mut();
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            cases.shuffle(&mut rng);

            driver.train(
                cases.iter(),
                AdamOptimizer {
                    a: 0.01,
                    b1: 0.9,
                    b2: 0.999,
                },
                32,
            );
        }

        let mut x = 0.0;
        while x < 1.0 {
            let abs_diff =
                ((x * f32::consts::TAU).sin() - driver.run(&[x]).output()[0] + 1.0).abs();
            assert!(abs_diff < 0.1, "the approximation must be accurate");
            x += 0.01;
        }
    }
}
