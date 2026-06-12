# C++ Neural Network Library

A lightweight, dependency-free Neural Network library written entirely in C++ from scratch. This project provides a flexible architecture to build, train, and test multi-layer perceptrons (MLPs). It includes support for various activation functions, loss functions, optimizers, and features a built-in data loader for the MNIST dataset.

## Features

* **Dynamic Architecture:** Easily define networks with any number of hidden layers and node sizes.
* **Activations:** ReLU, Linear.
* **Loss Functions:** Mean Squared Error (MSE), Softmax with Cross-Entropy (optimized for numerical stability).
* **Optimizers:** Stochastic Gradient Descent (SGD), Momentum SGD.
* **I/O Operations:** Save trained weights to a text file and load them for later inference.
* **No External Dependencies:** Only relies on the C++ Standard Library.

---

## 🚀 Getting Started

### 1. Clone the Repository
To get a local copy up and running, clone the repository using Git:
```bash
git clone https://github.com/AbdallahAbdalbasset/NeuralNetwork.git
cd NeuralNetwork
```

### 2. Compilation
Because the library heavily relies on matrix multiplications and loops, it is highly recommended to compile with optimization flags. Run the following command in your terminal:
```bash
g++ -O3 -march=native -ffast-math -std=c++17 NeuralNetwork.cpp -o out
```

### 3. Execution
Once compiled, you can run the pre-compiled executable (or the one you just built) using:
```bash
./out
```

---

## 🧠 Core Functions Explained

Understanding the underlying mechanics of the network is straightforward. The library relies on three main operations during the training loop:

### `forward(vector<double>& input)`
* **What it does:** Calculates the forward pass (predictions) of the network.
* **How it works:** 1. It takes an input vector and feeds it into the first layer.
  2. For each subsequent layer, it calculates the dot product of the previous layer's outputs and the connecting weights, then adds the bias. 
  3. This raw sum (`netVal`) is passed through the layer's activation function to produce the output (`outVal`).
  4. Crucially, these values are **stored inside each neuron**, which is required later for calculating gradients during backpropagation.

### `backProp(vector<double> target)`
* **What it does:** Computes the gradient (direction and magnitude of error) for every weight and bias in the network. 
* **How it works:**
  1. It requires a `target` vector that is the exact same size as the output layer.
  2. It starts at the final layer, using the selected `Loss` function to calculate the initial error between the network's output and the target.
  3. It then traverses backward through the network (Backpropagation), applying the chain rule. It calculates the derivative of the activation functions and distributes the error backwards to find exactly how much each weight and neuron contributed to the total loss.
  4. It **stores these gradients** inside each `Weight` and `Neuron` struct, but it **does not** apply the updates to the weights yet.

### `sgd.step()`
* **What it does:** Updates the actual network weights.
* **How it works:** 1. Now that `backProp` has calculated the gradients, the optimizer steps in. 
  2. Depending on the optimizer initialized (e.g., standard `SGD` or `MomentumSGD`), it iterates through every weight in the network.
  3. It subtracts a portion of the gradient (scaled by the learning rate) from the current weight value. If using Momentum, it also factors in the velocity of previous updates to accelerate training and avoid local minima.

---

## 💻 Usage Example

Here is a quick snippet demonstrating how to initialize and train a network using this library:

```cpp
#include "NeuralNetwork.h" // Assuming you separate into headers later

int main() {
    // 1. Define Network Architecture: 784 inputs, 256/128 hidden, 10 outputs
    NeuralNetwork nn({28*28, 256, 128, 10}, // Architecture 
                     new SoftmaxCrossEntropy(), // Loss function
                     {new Relu(), new Relu(), new Linear()}); // Vector of activations, Each layer can have a different Activation
    
    // 2. Initialize Optimizer (Momentum SGD with lr=0.001, momentum=0.9)
    MomentumSGD sgd(nn.weights, 0.001, 0.9);

    // 3. Training Loop (Simplified)
    vector<double> sampleInput = /* ... load input data ... */;
    vector<double> targetOutput = /* ... load one-hot target ... */;

    // Forward pass: calculate predictions
    nn.forward(sampleInput);

    // Backward pass: calculate gradients
    nn.backProp(targetOutput);

    // Optimizer step: apply weight updates
    sgd.step();

    // 4. Save model state
    nn.saveNNtoFile("Weights.txt");

    return 0;
}
```

## Dataset Note
To run the included `main()` function successfully, ensure you have the standard MNIST dataset files (`train-images-idx3-ubyte`, `train-labels-idx1-ubyte`, etc.) in the same directory as the executable.
