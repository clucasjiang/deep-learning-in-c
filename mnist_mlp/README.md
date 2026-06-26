This implements a simple MNIST multilayer perceptron (MLP) inference model in C.

PyTorch trains the model and provides the parameters. The C code only runs the forward pass.

Model architecture (from the pytorch quickstart tutorial):
input layer: 784
2 hidden layers of 512 each
output layer: 10
ReLU in between

The MNIST dataset is stored at the repo root under `mnist_dataset/`.
