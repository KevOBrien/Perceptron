import numpy as np

learningRate = 0.01
numInputs = 3
dataSetSize = 100
numEpochs = 100
printStep = 10

# Input: num_inputs numbers between -10 and +10
# Label: 1 if sum of inputs > 0, otherwise 0
inputs, labels = [], []
for i in range(dataSetSize):
    input = np.random.uniform(-10, 10, numInputs)
    inputs.append(input)
    labels.append((np.sum(input) > 0).astype(int))
