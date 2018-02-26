import tensorflow as tf
# import numpy as np


# Constants
in_dim = 8
learning_rate = 0.0001
hidden_sz = 100
batch_size = 32


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


loss = tf.sqrt(
    tf.reduce_mean(tf.squared_difference(Y, out_2)),
    name='root_mean_squared_error')
# loss = tf.reduce_mean(
    # tf.squared_difference(Y, out_2),
    # name='mean_squared_error')
tf.summary.scalar("loss", loss)


with tf.name_scope("correct-percent") as scope:
    # predictions = tf.round(tf.Print(out_2, [out_2]))
    predictions = tf.round(out_2)
    # correct = tf.count_nonzero(predictions == Y)
    correct = tf.reduce_sum(predictions)
    print(correct)
tf.summary.scalar("correct", correct)


# optimizer = tf.train.AdamOptimizer(learning_rate)
# optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
optimizer = tf.train.GradientDescentOptimizer(0.1)
train_step = optimizer.minimize(loss)

merged = tf.summary.merge_all()
