import numpy as np

import Data


class Perceptron:

    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = np.random.random((num_inputs, 1))
        self.bias = np.random.random()
        self.w_gradients = np.zeros_like(self.weights)      # Changes to be made to weights and bias
        self.b_gradient = np.zeros_like(self.bias)          # calculated through backprop

    def sigmoid(self, x, derivative=False):
        if derivative:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        self.inputs = np.array(inputs).reshape(self.num_inputs, 1)  # Python list -> Numpy array, (3,) -> (3, 1)
        weighted_sum = np.dot(inputs, self.weights) + self.bias     # xW + b
        self.prediction = self.sigmoid(weighted_sum)                # Activation
        return self.prediction

    def backward(self, label):
        loss = (label - self.prediction) ** 2   # Squared Error
        delta = 2 * (label - self.prediction) * self.sigmoid(self.prediction, derivative=True)  # Output delta
        self.w_gradients += self.inputs.dot(delta).reshape(self.num_inputs, 1)                  # Changes to be ade to weights
        self.b_gradient += delta[0]     # Change to be made to bias - (1,1) matrix -> single digit
        return loss

    def update(self, learning_rate):
        self.weights += learning_rate * self.w_gradients
        self.bias += learning_rate * self.b_gradient
        self.w_gradients = np.zeros_like(self.weights)      # Reset changes to zero after update
        self.b_gradient = np.zeros_like(self.bias)


perceptron = Perceptron(Data.num_inputs)

for i in range(Data.num_epochs):
    epoch_loss = 0
    epoch_correct = []
    for j in range(len(Data.inputs)):
        prediction = perceptron.forward(Data.inputs[j])
        loss = perceptron.backward(Data.labels[j])
        epoch_loss += loss
        epoch_correct.append(np.round(prediction) == Data.labels[j])
    perceptron.update(Data.learning_rate)
    if i == 0 or (i + 1) % Data.print_step == 0:
        print("Epoch:", i + 1)
        print("Error:", (epoch_loss / len(Data.inputs))[0])
        print("Acc  :", (np.sum(epoch_correct) / len(Data.inputs)) * 100, "%\n")
