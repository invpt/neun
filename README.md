# Neun
<img align="right" src="https://user-images.githubusercontent.com/57822954/225159983-7ae06e0e-ed6e-4943-8a5d-cc18fea25b43.svg" />
A flexible CPU-based neural network library for Rust.

## Features 
Currently, Neun only supports feed-forward networks with fully-connected layers using the ReLU activation function. If this fits your desired network topology, then you should be pretty happy with what Neun lets you do. :)

Stochastic gradient descent and Adam optimizers are included, but you can also write your own optimizer if desired.

## Simple example
The example below uses Neun to train a small network to approximate `sin(x)` for `0 ≤ x ≤ 2π`.
```rs
use std::f32::consts::{PI, TAU};

// create some training data
let mut cases = Vec::new();
for i in 0..10_000 {
    let x = (i as f32 / 10_000.0) * TAU;

    // add 1 to x.sin() so that it is
    // nonnegative since we're using the
    // ReLU activation function, which does
    // not support negative values
    cases.push(([x], [x.sin() + 1.0]));
}

// create the model to have one input
// neuron, a single 32-neuron hidden
// layer, and one output neuron
let mut model = Model::new(&[1, 32, 1]);

// get a driver to run the model with
let mut driver = model.driver_mut();

// repeatedly train over 100 epochs
for _ in 0..100 {
    // shuffle the input before each epoch,
    // since it's usually not a good idea
    // to train with sorted data or with
    // data in the same order every time
    cases.shuffle(&mut rand::thread_rng());

    // train with the shuffled dataset
    driver.train(
        cases.iter(),
        // use the Adam optimization
        // algorithm with some baseline
        // hyperparameters
        AdamOptimizer {
            a: 0.01,
            b1: 0.9,
            b2: 0.999,
        },
        // use a batch size of 32, which
        // means the model's weights are
        // adjusted every 32 cases
        32,
    );
}

// approximate sin(π) using the model
let sin_pi = driver.run(&[PI]).output()[0];

// check that the answer is accurate;
// we know that sin(π) should equal 0
assert!(sin_pi.abs() < 0.1);
```

## Custom training loop
I created Neun because I needed a library that allows full control over the training loop, and it achieves this goal. For example, the code below shows a slight simplification of the implementation of the `train` method that was used in the example above, notably missing a batch size for brevity.
```rs
let cases = /* some training data */;
let optimizer = /* some Optimizer */;
let mut model = /* some Model */;

// save the variable count to avoid recomputation
let variable_count = model.variable_count();

// create a driver
let mut driver = model.driver_mut();

// instantiate the optimizer to optimize all the
// variables in the current model
let mut optimizer = optimizer.instance(variable_count);

// create a vector to store the gradient
let mut dx = vec![0.0; variable_count];

for (input, target) in cases {
    // feed the network the input and then compute the gradient
    self.run_and_record(input)
        .compute_gradients(target, |idx, val| dx[idx] = val);

    // optimize the network
    optimizer.apply(driver.model_mut().variables_mut().zip(dx.iter()));
}
```

## Goals
Neun strives to be a useful and performant library for training and and using neural networks. While it currently only supports relatively simple network topologies, this may change in the future to support arbitrary topologies that fit into a directed acyclic graph. Neun also strives to be flexible, meaning that it hopes to support a wide variety of use cases for the supported topologies, such as custom training loops.

## Non-goals
Neun does not have plans to implement GPU-based inference or training; it is purely intended for running on the CPU.
