# MNIST MLP

This implements a MNIST multilayer perceptron (MLP) model in C.
See the [project overview](../README.md) for results, technical highlights, and a code guide.


Model architecture (from the pytorch quickstart tutorial):
input layer: 784
2 hidden layers of 512 each
output layer: 10
ReLU in between

The MNIST dataset is stored at the repo root under `mnist_dataset/`.

## Train and evaluate

Run these commands from the repo root (the directory with the Makefile):

```sh
make                 # Build the program with clang
make train           # Train on all 60,000 training images
make eval            # Rebuild with the saved weights and test on 10,000 images
make run INDEX=100   # Predict one test image and show its actual label
```

Training starts from random weights each time. It makes 20 passes (epochs) through
the training set, with a learning rate of 0.01 and batches of 50 images. Each
epoch shuffles the images and prints the average training loss. This uses plain
C loops on the CPU, so an epoch may take a while.

To change the training settings, edit `epochs` and `learning_rate` in
`train_model` in `mlp.c`, or `BATCH_SIZE` in `mlp.h`, then run `make train`.

For each batch, backprop adds up the weight and bias gradients for all three
layers. ReLU blocks the gradient where its input was zero or negative. After the
whole batch, the program averages the gradients and subtracts the learning rate
times each average from its weight or bias.

Training overwrites `mnist_mlp/stored_weights.c` and `mnist_mlp/stored_weights.h`.
The weights are part of the compiled program, so **rebuild after training** before
running predictions or evaluation. `make eval` and `make run` do this for you.
Calling `./bin/mnist_mlp eval` directly before rebuilding uses the old weights.

Evaluation prints accuracy (the percentage of correct predictions) and average
cross-entropy loss on the separate test set. Lower loss is better; evaluation
does not change the weights.
