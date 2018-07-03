import numpy as np
import tensorflow as tf

import Data

x = tf.placeholder(tf.float32, [1, Data.numInputs])
y = tf.placeholder(tf.float32, [1])

weights = tf.Variable(tf.random_normal([Data.numInputs, 1]))
bias = tf.Variable(tf.random_normal([1]))

weightedSum = tf.matmul(x, weights) + bias
prediction = tf.sigmoid(weightedSum)[0]

loss = tf.losses.mean_squared_error(y, prediction)

train = tf.train.AdamOptimizer(Data.learningRate).minimize(loss)

with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    for i in range(Data.numEpochs):
        epochLoss = 0
        epochCorrect = []
        for j in range(len(Data.inputs)):
            _, l, pred, label = session.run([train, loss, prediction, y], {x: [Data.inputs[j]], y: [Data.labels[j]]})
            epochLoss += l
            epochCorrect.append(np.round(pred) == label)
        if i == 0 or (i + 1) % Data.printStep == 0:
            print("Epoch:", i + 1)
            print("Error:", (epochLoss / len(Data.inputs)))
            print("Acc  :", (np.sum(epochCorrect) / len(Data.inputs)) * 100, "%\n")
