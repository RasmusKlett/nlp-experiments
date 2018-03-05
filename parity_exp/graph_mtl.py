import tensorflow as tf
# import numpy as np


# Constants
in_dim = 8
learning_rate = 0.001
hidden_sz = 100


# ======================
# Graph definition
# ======================

X = tf.placeholder("float", [None, in_dim], name="X")
Y1 = tf.placeholder("float", [None], name="Y1")
Y2 = tf.placeholder("float", [None], name="Y2")
Y3 = tf.placeholder("float", [None], name="Y3")
Y4 = tf.placeholder("float", [None], name="Y4")


with tf.name_scope("shared_layer") as scope:
    W1 = tf.get_variable(
        "weights1",
        [in_dim, hidden_sz],
        # initializer=tf.zeros_initializer,
    )
    B1 = tf.get_variable("bias1", [hidden_sz])

    out_1 = tf.nn.relu(tf.matmul(X, W1) + B1)

with tf.name_scope("layer_t1") as scope:
    Wt1 = tf.get_variable(
        "weights_t1", [hidden_sz, 1],
        # initializer=tf.zeros_initializer
    )

    Bt1 = tf.get_variable("bias_t1", [1])
    out_t1 = tf.nn.relu(tf.matmul(out_1, Wt1) + Bt1)

with tf.name_scope("layer_t2") as scope:
    Wt2 = tf.get_variable(
        "weights_t2", [hidden_sz, 1],
        # initializer=tf.zeros_initializer
    )

    Bt2 = tf.get_variable("bias_t2", [1])
    out_t2 = tf.nn.relu(tf.matmul(out_1, Wt2) + Bt2)

loss_t1 = tf.losses.mean_squared_error(Y1, tf.reshape(out_t1, [-1]))
loss_t2 = tf.losses.mean_squared_error(Y2, tf.reshape(out_t2, [-1]))
loss = loss_t1 + loss_t2
tf.summary.scalar("loss_t1", loss_t1)
tf.summary.scalar("loss_t2", loss_t2)
tf.summary.scalar("loss", loss)


with tf.name_scope("correct-percent") as scope:
    predictions = tf.reshape(tf.round(out_t1), [-1])
    correct1 = tf.equal(predictions, Y1)
    correct = tf.to_float(tf.count_nonzero(correct1)) / \
        tf.to_float(tf.shape(out_t1)[0])

tf.summary.scalar("correct", correct)


# optimizer = tf.train.AdamOptimizer(learning_rate)
# optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
optimizer = tf.train.GradientDescentOptimizer(0.01)
train_step = optimizer.minimize(loss)

merged = tf.summary.merge_all()
