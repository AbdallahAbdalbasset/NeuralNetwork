# C++ Neural Network from Scratch

A lightweight, purely object-oriented Artificial Neural Network implemented in C++ from scratch. This implementation uses Stochastic Gradient Descent (SGD) for training and is designed to be easily extensible with custom loss functions and activation layers.

## Initialization

To instantiate the neural network, use the `NeuralNetwork` constructor:

```cpp
NeuralNetwork nn(layersSize, loss_function, activation_function, learning_rate);
```

### Constructor Parameters

| Parameter | Type | Description |
| :--- | :--- | :--- |
| `layersSize` | `vector<int>` | Defines the architecture of the network (Input -> Hidden -> Output). Example: `{2, 4, 5, 1}` creates a network with 2 input nodes, 4-nodes hidden layer, 5-nodes hidden layer and automatically appends a 1-node output layer if it is not explicitly provided. 
Each layer is 'fully connected' with the next one
Each layer has a single bias connceted to all neurons of the next one |
| `loss` | `Loss*` | A pointer to the loss function object (e.g., `new MSE()`). |
| `activation` | `Activation*` | A pointer to the activation function object (e.g., `new Relu()` or `new Linear()`). |
| `lr` | `double` | The learning rate for weight updates during Stochastic Gradient Descent. |

---

## Methods

### `forward`
```cpp
double forward(vector<double> input)
```
Performs the forward propagation step through the network.
* **Input:** A `std::vector<double>` representing the features of a single sample. Its size must strictly match the first element of `layersSize`.
* **Returns:** The final predicted `double` value from the output neuron.

### `backProp`
```cpp
void backProp(double target)
```
Performs backpropagation using Stochastic Gradient Descent (SGD). It **must** be called immediately after a `forward()` pass.
* **Input:** A `double` representing the true target value for the last processed sample.
* **Behavior:** Compares the previous forward pass output to the target, computes the loss gradient, calculates gradients for all layers, and immediately updates the weights.

---

## Example Usage: Learning the AND Gate

Here is a complete example of how to initialize, train, and test the network using a simple dataset.

```cpp
#include "bits/stdc++.h"
// ... (include the provided classes here)

int main() {
    // 1. Prepare the Data (AND Logic Gate)
    vector<vector<double>> X = {{0, 0}, {0, 1}, {1, 0}, {1, 1}};
    vector<double> Y = {0, 0, 0, 1};

    // 2. Create Neural Network
    // 2 inputs, 1 output. Mean Squared Error loss, ReLU activation, 0.01 learning rate.
    NeuralNetwork nn({2, 1}, new MSE(), new Relu(), 0.01);

    // 3. Training Loop (Stochastic Gradient Descent)
    srand(1);
    for(int i = 0; i < 40000; i++) {
        int idx = rand() % 4;        // Pick a random sample
        nn.forward(X[idx]);          // Forward pass
        nn.backProp(Y[idx]);         // Compute gradients and update weights
    }

    // 4. Test the Network
    for(int i = 0; i < 4; i++) {
        cout << "Sample: " << i + 1 << endl;
        cout << "Data: (" << X[i][0] << ", " << X[i][1] << ")   Target: " << Y[i] 
             << "   Prediction: " << nn.forward(X[i]) << endl;
    }

    return 0;
}
```
