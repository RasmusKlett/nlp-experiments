import tensorflow as tf
import numpy as np
from batcher import Batcher
from generate_bits import bits_prepared

# Data loading
X_full, Y_full = bits_prepared()
random_order = np.random.permutation(X_full.shape[0])

X_train, X_test = np.split(
        np.take(X_full, random_order, axis=0, out=X_full), 2)

Y_train, Y_test = np.split(
        np.take(Y_full, random_order, axis=0, out=Y_full), 2)

print("Y train trues:", sum(Y_train))
print("Y test trues:", sum(Y_test))

# Constants
max_iter = 1000000
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

    out_1 = tf.nn.sigmoid(tf.matmul(X, W1) + B1)

with tf.name_scope("layer2") as scope:
    W2 = tf.get_variable("weights2", [hidden_sz, 1])
    B2 = tf.get_variable("bias2", [1])
    out_2 = tf.nn.sigmoid(tf.matmul(out_1, W2))


loss = tf.sqrt(tf.reduce_mean(tf.squared_difference(Y, out_2), name='mean_squared_error'))
tf.summary.scalar("loss", loss)


with tf.name_scope("correct-percent") as scope:
    # predictions = tf.round(tf.Print(out_2, [out_2]))
    predictions = tf.round(out_2)
    # correct = tf.count_nonzero(predictions == Y)
    correct = tf.reduce_sum(predictions)
    print(correct)
tf.summary.scalar("correct", correct)


optimizer = tf.train.AdamOptimizer(learning_rate)
# optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
train_step = optimizer.minimize(loss)

merged = tf.summary.merge_all()


with tf.Session() as sess:  # set up the session
    sess.run(tf.global_variables_initializer())

    batcher = Batcher(X_train, Y_train, batch_size)
    x_batch, y_batch = batcher.nextBatch()

    writer = tf.summary.FileWriter("/tmp/log/parity/train", sess.graph)

    test_writer = tf.summary.FileWriter("/tmp/log/parity/test", sess.graph)

    for i in range(max_iter):
        summary, model_loss, _ = sess.run(
            [merged, loss, train_step],
            {
              X: x_batch,
              Y: y_batch,
            })
        writer.add_summary(summary, i)

        test_summary = sess.run(merged, {X: X_test, Y: Y_test})
        test_writer.add_summary(test_summary, i)
        if i % 20 == 0:
            print("Iteration %i    loss: %f" % (i, model_loss))
    writer.close()
