# Deep Learning in C

A handwritten digit classifier with forward propagation, backpropagation, and training implemented directly in C. The current model achieves **97.36% accuracy on the 10,000-image MNIST test set** using the saved weights included in this project.

This project makes the mechanics of a neural network visible: matrix operations, activation functions, gradients, and weight updates are expressed as readable loops over arrays. Training and inference run on the CPU using the C standard library and math library, with no machine learning framework required.

## Technical highlights

- **Manual backpropagation:** computes weight and bias gradients for all three fully connected layers, including the ReLU derivatives.
- **Batch training:** shuffles the training set each epoch, accumulates gradients across a batch, and updates parameters using their average.
- **Numerical care:** subtracts the largest logit before softmax exponentiation and adds a small epsilon when computing cross-entropy loss.
- **Binary data handling:** reads MNIST images and labels directly from IDX files and scales pixel values to the range `[0, 1]`.
- **Explicit memory layout:** uses fixed-size arrays for parameters, cached activations, and gradients; the large gradient buffer uses static storage to avoid placing it on the stack.
- **Weight persistence:** exports trained parameters as C source arrays that are compiled into the prediction program.

## Model

The multilayer perceptron (MLP) follows the architecture in the PyTorch quickstart tutorial, with the computation and training implemented in C.

```text
28 × 28 grayscale image
          ↓ flatten and scale
      784 inputs
          ↓ fully connected + ReLU
      512 neurons
          ↓ fully connected + ReLU
      512 neurons
          ↓ fully connected + softmax
  10 digit probabilities (0–9)
```

The network has **669,706 trainable parameters**. It uses cross-entropy loss and mini-batch stochastic gradient descent (SGD).

## Evaluation

Measured locally with `make eval` on September 15, 2026:

| Metric | Result |
| --- | ---: |
| Test accuracy | **97.36%** |
| Correct predictions | 9,736 / 10,000 |
| Average test loss | 0.0888 |

These results describe the current saved weights. The evaluation command uses the separate MNIST test split and does not update the model. Retraining produces a new set of weights whose results can be checked with the same command.

## Quick start

Requirements: `make`, a C11 compiler (`clang` by default), and the MNIST data files under `mnist_dataset/MNIST/raw/`.

Run from the repository root:

```sh
make                 # Build the program
make eval            # Evaluate the saved model on all 10,000 test images
make run INDEX=100   # Show a prediction, confidence, and actual digit
```

`INDEX` selects a test image from `0` to `9999`. To use GCC, build with `make CC=gcc`.

The data reader expects these uncompressed files:

```text
mnist_dataset/MNIST/raw/
├── train-images-idx3-ubyte
├── train-labels-idx1-ubyte
├── t10k-images-idx3-ubyte
└── t10k-labels-idx1-ubyte
```

## Train from scratch

```sh
make train
make eval
```

Current training settings:

| Setting | Value |
| --- | --- |
| Training images | 60,000 |
| Epochs | 20 |
| Batch size | 50 |
| Learning rate | 0.01 |
| Initialization seed | 1234 |

Each epoch prints the average training loss. Change `epochs` and `learning_rate` in [`train_model`](mnist_mlp/mlp.c), or `BATCH_SIZE` in [`mlp.h`](mnist_mlp/mlp.h), to experiment.

**Each training run starts from random weights and overwrites `stored_weights.c` and `stored_weights.h`.** Saved weights are compiled into the executable, so it must be rebuilt after training. `make eval` and `make run` handle that rebuild automatically.

## Code guide

| File | Responsibility |
| --- | --- |
| [`mlp.c`](mnist_mlp/mlp.c) | Forward pass, loss, backpropagation, parameter updates, and training loop |
| [`mlp.h`](mnist_mlp/mlp.h) | Network dimensions, activation cache, and gradient arrays |
| [`main.c`](mnist_mlp/main.c) | Command-line interface, individual predictions, and test-set evaluation |
| [`read_image.c`](mnist_mlp/read_image.c) | MNIST image and label loading |
| [`mlp_weights.c`](mnist_mlp/mlp_weights.c) | Parameter initialization, loading, and export |
| [`stored_weights.c`](mnist_mlp/stored_weights.c) | Saved model parameters |
| [`Makefile`](Makefile) | Build, train, predict, and evaluate commands |

## Scope and next steps

The implementation prioritizes readable C and explicit control over the training process. It currently uses a single CPU thread, a fixed learning rate, and plain SGD. Useful extensions include validation-based checkpoint selection, momentum, learning-rate scheduling, and faster matrix operations.
