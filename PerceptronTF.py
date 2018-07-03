import numpy as np
import tensorflow as tf

import Data

x = tf.placeholder(tf.float32, [1, Data.num_inputs])
y = tf.placeholder(tf.float32, [1])

weights = tf.Variable(tf.random_normal([Data.num_inputs, 1]))
bias = tf.Variable(tf.random_normal([1]))

weighted_sum = tf.matmul(x, weights) + bias
prediction = tf.sigmoid(weighted_sum)[0]

loss = tf.losses.mean_squared_error(y, prediction)

train = tf.train.AdamOptimizer(Data.learning_rate).minimize(loss)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(Data.num_epochs):
        epoch_loss = 0
        epoch_correct = []
        for j in range(len(Data.inputs)):
            _, l, pred, label = session.run([train, loss, prediction, y], {x: [Data.inputs[j]], y: [Data.labels[j]]})
            epoch_loss += l
            epoch_correct.append(np.round(pred) == label)
        if i == 0 or (i + 1) % Data.print_step == 0:
            print("Epoch:", i + 1)
            print("Error:", (epoch_loss / len(Data.inputs)))
            print("Acc  :", (np.sum(epoch_correct) / len(Data.inputs)) * 100, "%\n")
