# MNIST from scratch in C++ #

This repository contains a simple implementation of a neural network trained on the MNIST dataset using C++. 
The code demonstrates how to build and train a neural network from scratch without relying on high-level libraries.

## Features ##

| Feature             | Description                                      | Status  |
|---------------------|--------------------------------------------------|---------|
| Data Loading        | Load MNIST dataset from files                    | ✅       |
| Linear Layer        | Simple feedforward neural network implementation | ✅       |
| Activation          | ReLU and Softmax activation functions            | ✅       |
| Loss Function       | Cross-entropy loss calculation                   | ✅       |
| Optimizer           | Stochastic Gradient Descent (SGD) optimizer      | ✅     |
| Backpropagation     | Backpropagation as part of model                 | ✅     |
| Evaluation          | Evaluate model accuracy on test set              | ✅       |
| Visualization       | Visualize training progress and results          | WIP     |
| Trainer             | Training loop management                         | ✅       |
| Batched Training    | Support for mini-batch training                  | Planned |
| Graph Based Linking | Modular layer connections                        | Planned |
| DB Logging         | Log training metrics to a database               | Planned |