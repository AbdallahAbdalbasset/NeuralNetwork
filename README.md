# C++ Neural Network from Scratch

A lightweight, purely object-oriented, and highly modular Artificial Neural Network implemented in C++ from scratch with zero external dependencies. Designed for efficiency and extensibility, this framework cleanly separates components into decoupled layers, custom activation functions, loss abstractions, and advanced optimization algorithms.

The included driver application demonstrates the network's capabilities by training on the classic **MNIST Handwritten Digit Dataset**, achieving high-performance execution via aggressive compiler optimizations.

---

## Key Features

* **Modular Components:** Easily implement and swap custom activation functions, loss metrics, or optimization strategies via clean abstract base classes.
* **Advanced Optimization:** Includes standard Stochastic Gradient Descent (SGD) and **Momentum-based SGD** to accelerate convergence and navigate tricky error surfaces.
* **Numerical Stability:** The `SoftmaxCrossEntropy` loss implements the log-sum-exp trick implicitly during propagation to safely handle potential floating-point overflow/underflow.
* **Serialization:** Full built-in support to save trained network state models (`saveNNtoFile`) and hot-load weights (`loadFromFile`) later for inference.
* **MNIST Pipeline Built-in:** Includes a raw binary parser for IDX-formatted datasets and a PGM image exporter to visually audit dataset contents.

---

## Project Architecture

The library splits network components into clear, intuitive data structures and polymorphic classes:

### Core Data Layout
* **`Weight` Struct:** Tracks its current value, raw analytical gradient, and historical velocity vector (crucial for momentum calculations).
* **`Neuron` Struct:** Tracks pre-activation sums (`netVal`), post-activation outputs (`outVal`), and node-level local error gradients (`grad`).

### Class Hierarchy
| Component | Supported Types | Description |
| :--- | :--- | :--- |
| **`Activation`** | `Linear`, `Relu` | Intercepts node-level sums during forward and backward passes. |
| **`Loss`** | `MSE`, `SoftmaxCrossEntropy` | Calculates scalar loss evaluations and seeds final layer error configurations. |
| **`Optimizer`** | `SGD`, `MomentumSGD` | Walks the parameter space using decoupled weight/gradient references. |

---

## Quick Start & Usage

### 1. Prerequisites
Before running, ensure you have downloaded the raw MNIST dataset files and placed them directly into the root folder of your project workspace:
* `train-images-idx3-ubyte` (Training Images)
* `train-labels-idx1-ubyte` (Training Labels)
* `t10k-images-idx3-ubyte` (Testing Images)
* `t10k-labels-idx1-ubyte` (Testing Labels)

### 2. Setup Options

#### Option A: Running the Pre-compiled Binary
If you are using the pre-compiled executable provided inside the repository workspace, you can execute it immediately:

```bash
# Give it execution privileges if necessary (Linux/macOS)
chmod +x out

# Execute the pre-compiled binary
./out
```

#### Option B: Compiling from Source
To compile the implementation files yourself, use the high-performance compilation string optimized for maximum vectorization and hardware optimization:

```bash
g++ -O3 -march=native -ffast-math -std=c++17 NeuralNetwork.cpp -o out
./out
```

---

## Network Instantiation & Example

The following code details how to structure a network topology using your custom architectural parameters, train it on data matrices, and output model checkpoints:

```cpp
// 1. Define Topology: 784 Inputs (28x28 pixels), two hidden layers, and 10 Output classes
vector<int> topology = {784, 256, 128, 10};

// 2. Map Activation instances per layer transition (3 transitions total)
vector<Activation*> activations = {new Relu(), new Relu(), new Linear()};

// 3. Instantiate Network shell with Loss criteria
NeuralNetwork nn(topology, new SoftmaxCrossEntropy(), activations);

// 4. Attach Optimizer tracking parameters directly from the network instance
MomentumSGD optimizer(nn.weights, 0.001, 0.9);

// --- Training Loop Iteration ---
for (int i = 0; i < trainData.size(); i++) {
    // Forward propagation step
    nn.forward(trainData[i].features);
    
    // Backpropagation pass calculates local gradients
    nn.backProp(trainData[i].oneHotLabel);
    
    // Optimizer walks weights down the loss gradient curve
    optimizer.step();
}

// 5. Serialize model state to disk
nn.saveNNtoFile("Weights.txt");
```

---

## Model Serialization Model File Formats
When invoking `saveNNtoFile("filename.txt")`, the network saves configuration dimensions and continuous raw weights inside a lightweight structured text schema:

```text
<total_layers_count>
<layer_0_size> <layer_1_size> ... <layer_n_size> 
<raw_weight_value_1>
<raw_weight_value_2>
...
```
This enables rapid data reconstruction later via `NeuralNetwork::loadFromFile("Weights.txt", loss, activations)`.
