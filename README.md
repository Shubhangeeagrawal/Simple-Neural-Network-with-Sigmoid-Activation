# Simple-Neural-Network-with-Sigmoid-Activation

### Overview
This project implements a single-layer neural network (perceptron) in Python to demonstrate the basics of neural network training.Using numpy, this project performs binary classification with a simple dataset and trains the neural network through forward propagation, error calculation, and backpropagation.

This example provides a foundational understanding of how neural networks learn by adjusting weights to minimize prediction errors. The project is particularly useful for those new to neural networks, as it demonstrates core concepts without the complexity of multiple layers.

### Theory
This single-layer neural network is a basic form of an artificial neural network designed for binary classification. The main components and concepts are as follows:

**1. Sigmoid Activation Function:** The sigmoid function maps any input into a range between 0 and 1, making it useful for binary classification. It is defined as:
                                  **σ(x)= 1/1+e^(−x)** 
Its derivative is used during backpropagation to update weights, allowing the model to learn from errors.

**2. Data and Labels:**
- Input data (X): Each row represents a data sample with binary feature values.
- Target labels (y): The expected outputs for each input sample, which are binary values in this case.

**3. Weight Initialization:** Weights are initialized with small random values to break symmetry and provide a starting point for learning. The weights determine how input data is transformed through the network.

**4. Forward Propagation:** The input data is multiplied by the weights and passed through the sigmoid function to produce predictions. This step is essential in calculating the output for each sample based on current weights.

**5. Error Calculation:** The difference between the predicted and actual outputs is computed as the error. This error guides how the weights are adjusted during backpropagation.

**6. Backpropagation and Weight Update:** The error is used to calculate how much each weight needs to change. The network calculates adjustments by multiplying the error with the derivative of the sigmoid function, which helps minimize the error over many iterations.

## Code Explanation
_import numpy as np_
- numpy: Library used for efficient mathematical operations on arrays, making it ideal for handling data and weight matrices in neural networks.

### Sigmoid Activation Function
_def nonlin(x, deriv=False):
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))_
- nonlin function: Represents the sigmoid function.
  - deriv=True calculates the derivative of the sigmoid function, used in backpropagation.
  - deriv=False calculates the sigmoid output, used in forward propagation.

### Input and Output Data
_X = np.array([
    [0, 0, 1],
    [0, 1, 1],
    [1, 0, 1],
    [1, 1, 1]
])
y = np.array([[0, 0, 1, 1]]).T_
- X: Input data where each row is a sample with binary features.
- y: Target labels for each sample, representing the desired output.

### Weight Initialization
_np.random.seed(1)
syn0 = 2 * np.random.random((3, 1)) - 1_
-syn0: Initial weight matrix, randomly set to values between [−1,1]. The matrix has dimensions (3, 1), matching the number of features in X.

### Training Loop
_for iter in range(10000):
    l0 = X
    l1 = nonlin(np.dot(l0, syn0))
    l1_error = y - l1
    l1_delta = l1_error * nonlin(l1, True)
    syn0 += np.dot(l0.T, l1_delta)_
1. l0: Represents the input layer (i.e., the input data X).
2. l1: The predicted output from the neural network after applying the sigmoid function to the weighted input.
3. l1_error: The difference between the target output (y) and the predicted output (l1), representing the error.
4. l1_delta: Adjustment to be made to weights, calculated as l1_error multiplied by the derivative of the sigmoid function.
5. syn0 += np.dot(l0.T, l1_delta): Updates weights by applying the calculated adjustment (l1_delta). Each iteration improves the weights to reduce error over time.

### Output After Training
_print("Output After Training:")
print(l1)_
After 10,000 iterations, the neural network should have learned weights that allow it to make accurate predictions close to the target values.

### Summary of Variables
- X: Input dataset.
- y: Target output.
- syn0: Weight matrix between input and output layers.
- l0: Input layer (equivalent to X).
- l1: Predicted output after applying the sigmoid function.
- l1_error: Error between predicted and actual outputs.
- l1_delta: Adjustments to weights to minimize error.

### Conclusion
This project demonstrates the fundamental process of training a neural network by updating weights iteratively through forward propagation, error calculation, and backpropagation. This foundational code can be expanded to multiple layers and complex activation functions to solve more sophisticated tasks.
