import numpy as np

learning_rate = 0.01
num_inputs = 3
data_set_size = 100
num_epochs = 100
print_step = 10


class Perceptron:

    def __init__(self, num_inputs):
        self.num_inputs = num_inputs
        self.weights = np.random.random((num_inputs, 1))
        self.bias = np.random.random()
        self.w_gradients = np.zeros_like(self.weights)
        self.b_gradient = np.zeros_like(self.bias)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward(self, inputs):
        self.inputs = np.array(inputs).reshape(self.num_inputs, 1)
        weighted_sum = np.dot(inputs, self.weights) + self.bias
        self.prediction = self.sigmoid(weighted_sum)
        return self.prediction

    def backward(self, label):
        loss = (label - self.prediction) ** 2
        delta = 2 * (label - self.prediction) * self.sigmoid_derivative(self.prediction)
        self.w_gradients += self.inputs.dot(delta).reshape(self.num_inputs, 1)
        self.b_gradient += delta[0]
        return loss

    def update(self, learning_rate):
        self.weights += learning_rate * self.w_gradients
        self.bias += learning_rate * self.b_gradient
        self.w_gradients = np.zeros_like(self.weights)
        self.b_gradient = np.zeros_like(self.bias)


# Input: num_inputs numbers between -10 and +10
# Label: 1 if sum of inputs > 0, otherwise 0
inputs, labels = [], []
for i in range(data_set_size):
    input = np.random.uniform(-10, 10, num_inputs)
    inputs.append(input)
    labels.append((np.sum(input) > 0).astype(int))

perceptron = Perceptron(num_inputs)

for i in range(num_epochs):
    epoch_loss = 0
    epoch_correct = []
    for j in range(len(inputs)):
        prediction = perceptron.forward(inputs[j])
        loss = perceptron.backward(labels[j])
        epoch_loss += loss
        epoch_correct.append(np.round(prediction) == labels[j])
    perceptron.update(learning_rate)
    if i == 0 or (i + 1) % print_step == 0:
        print("Epoch:", i + 1)
        print("Error:", (epoch_loss / len(inputs))[0])
        print("Acc  :", (np.sum(epoch_correct) / len(inputs)) * 100, "%\n")
