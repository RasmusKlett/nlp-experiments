import tensorflow as tf
import numpy as np


# Constants
in_dim = 8
learning_rate = 0.001
hidden_sz = 100


# ======================
# Graph definition
# ======================

X = tf.placeholder("float", [None, in_dim], name="X")
Y = tf.placeholder("float", [None], name="Y")


with tf.name_scope("layer1") as scope:
    W1 = tf.get_variable("weights1", [in_dim, hidden_sz])
    B1 = tf.get_variable("bias1", [hidden_sz])

    out_1 = tf.nn.relu(tf.matmul(X, W1) + B1)

with tf.name_scope("layer2") as scope:
    W2 = tf.get_variable("weights2", [hidden_sz, 1])
    B2 = tf.get_variable("bias2", [1])
    out_2 = tf.nn.relu(tf.matmul(out_1, W2) + B2)


loss = tf.losses.mean_squared_error(Y, tf.reshape(out_2, [-1]))
tf.summary.scalar("loss", loss)


with tf.name_scope("correct-percent") as scope:
    predictions = tf.reshape(tf.round(out_2), [-1])
    # print(predictions)
    # print(Y)
    correct1 = tf.equal(predictions, Y)
    print(correct1)
    correct = tf.to_float(tf.count_nonzero(correct1)) / tf.to_float(tf.shape(out_2)[0])
    # tf.Print(None, (predictions == Y).shape)
    # correct = tf.reduce_sum(predictions)
tf.summary.scalar("correct", correct)


# optimizer = tf.train.AdamOptimizer(learning_rate)
# optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)

merged = tf.summary.merge_all()
