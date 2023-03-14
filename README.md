# Neun
<img align="right" src="https://user-images.githubusercontent.com/57822954/225159983-7ae06e0e-ed6e-4943-8a5d-cc18fea25b43.svg" />
A flexible CPU-based neural network library for Rust.

## Features 
Currently, Neun only supports feed-forward networks with fully-connected layers using the ReLU activation function. If this fits your desired network topology, then you should be pretty happy with what Neun lets you do. :)

Stochastic gradient descent and Adam optimizers are included, but you can also write your own optimizer if desired.

## Example
The example below uses Neun to train a small network to approximate `sin(x)` for `0 ≤ x ≤ 2π`.
```rs
use std::f32::consts::{PI, TAU};

// create some test data
let mut cases = Vec::new();
for i in 0..10_000 {
    let x = (i as f32 / 10_000.0) * TAU;

    // add 1 to x.sin() so that it is nonnegative
    // since we're using the ReLU activation function,
    // which does not support negative values
    cases.push(([x], [x.sin() + 1.0]));
}

// create the model to have one input neuron, a single
// 32-neuron hidden layer, and one output neuron
let mut model = Model::new(&[1, 32, 1]);

// create a driver to run and train the model
let mut driver = model.driver_mut();

// train the model over 200 epochs
for _ in 0..100 {
    // shuffle the input before each epoch,
    // since it's usually not a good idea to train
    // with sorted data or with data in the same
    // order every time
    cases.shuffle(&mut rand::thread_rng());

    // train the model with the shuffled dataset
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

// calculate the approximate value of sin(pi) using the network
let sin_pi = driver.run(&[PI]).output()[0];

assert!(sin_pi.abs() < 0.1, "should be accurate to one decimal point");
```

## Goals and non-goals
Neun strives to be a useful and performant library for training and and using neural networks. While it currently only supports relatively simple network topologies, this may change in the future to support arbitrary topologies that fit into a directed acyclic graph.
