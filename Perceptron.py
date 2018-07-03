import numpy as np

import Data


class Perceptron:

    def __init__(self, numInputs):
        self.numInputs = numInputs
        self.weights = np.random.random((numInputs, 1))
        self.bias = np.random.random()
        self.wGradients = np.zeros_like(self.weights)      # Changes to be made to weights and bias
        self.bGradient = np.zeros_like(self.bias)          # calculated through backprop

    def sigmoid(self, x, derivative=False):
        if derivative:
            return x * (1 - x)
        return 1 / (1 + np.exp(-x))

    def forward(self, inputs):
        self.inputs = np.array(inputs).reshape(self.numInputs, 1)  # Python list -> Numpy array, (3,) -> (3, 1)
        weightedSum = np.dot(inputs, self.weights) + self.bias     # xW + b
        self.prediction = self.sigmoid(weightedSum)                # Activation
        return self.prediction

    def backward(self, label):
        loss = (label - self.prediction) ** 2   # Squared Error
        delta = 2 * (label - self.prediction) * self.sigmoid(self.prediction, derivative=True)  # Output delta
        self.wGradients += self.inputs.dot(delta).reshape(self.numInputs, 1)                    # Changes to be made to weights
        self.bGradient += delta[0]     # Change to be made to bias - (1,1) matrix -> single digit
        return loss

    def update(self, learningRate):
        self.weights += learningRate * self.wGradients
        self.bias += learningRate * self.bGradient
        self.wGradients = np.zeros_like(self.weights)      # Reset changes to zero after update
        self.bGradient = np.zeros_like(self.bias)


perceptron = Perceptron(Data.numInputs)

for i in range(Data.numEpochs):
    epochLoss = 0
    epochCorrect = []
    for j in range(len(Data.inputs)):
        prediction = perceptron.forward(Data.inputs[j])
        loss = perceptron.backward(Data.labels[j])
        epochLoss += loss
        epochCorrect.append(np.round(prediction) == Data.labels[j])
    perceptron.update(Data.learningRate)
    if i == 0 or (i + 1) % Data.printStep == 0:
        print("Epoch:", i + 1)
        print("Error:", (epochLoss / len(Data.inputs))[0])
        print("Acc  :", (np.sum(epochCorrect) / len(Data.inputs)) * 100, "%\n")
