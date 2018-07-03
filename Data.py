import numpy as np

learning_rate = 0.01
num_inputs = 3
data_set_size = 100
num_epochs = 100
print_step = 10

# Input: num_inputs numbers between -10 and +10
# Label: 1 if sum of inputs > 0, otherwise 0
inputs, labels = [], []
for i in range(data_set_size):
    input = np.random.uniform(-10, 10, num_inputs)
    inputs.append(input)
    labels.append((np.sum(input) > 0).astype(int))
