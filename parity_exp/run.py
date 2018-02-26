import numpy as np
import tensorflow as tf
import itertools

from generate_bits import bits_prepared
from graph import X, Y, W1, W2, B1, B2, out_1, out_2, merged, loss, train_step


# Constants
max_iter = 50000


# Data loading
# X_full = np.round(np.random.rand(256, 8))
# Y_full = np.round(np.random.rand(256))
# X_train, X_test = np.split(X_full, 2)
# Y_train, Y_test = np.split(Y_full, 2)

X_full, Y_full = bits_prepared()
X_full = np.array(X_full, dtype=float)
Y_full = np.array(Y_full, dtype=float)

random_order = np.random.permutation(X_full.shape[0])

X_train, X_test = np.split(
        np.take(X_full, random_order, axis=0, out=X_full), 2)

Y_train, Y_test = np.split(
        np.take(Y_full, random_order, axis=0, out=Y_full), 2)

y_mean = np.mean(Y_train)

print(y_mean)

# Perform oversampling of False datapoints
# falses = X_train[~Y_train]
# count_false = len(falses)
# count_true = len(Y_train[Y_train])

# extra_count = count_true - count_false
# extras = []

# for i in range(extra_count):
    # extras.append(X_train[i, :])

# extras = np.array(extras)


# # Add replicated data to training set
# X_train = np.concatenate((X_train, extras), axis=0)
# Y_train = np.concatenate(
    # (Y_train, list(itertools.repeat(False, extra_count))),
    # axis=0)

# random_order_train = np.random.permutation(X_train.shape[0])
# X_train = np.take(X_train, random_order_train, axis=0, out=X_train)
# Y_train = np.take(Y_train, random_order_train, axis=0, out=Y_train)

# print("Y train trues:", sum(Y_train))
# print("Y test trues:", sum(Y_test))


# Run session
sess = tf.Session()

sess.run(tf.global_variables_initializer())

# batcher = Batcher(X_train, Y_train, batch_size)
# x_batch, y_batch = batcher.nextBatch()
x_batch = X_train  # Currently no batching
y_batch = Y_train

writer = tf.summary.FileWriter("/tmp/log/parity/train", sess.graph)

test_writer = tf.summary.FileWriter("/tmp/log/parity/test", sess.graph)

for i in range(max_iter):
    summary, model_loss, _, debug = sess.run(
        [merged, loss, train_step, out_2],
        {
          X: x_batch,
          Y: y_batch,
        })
    writer.add_summary(summary, i)

    test_summary = sess.run(merged, {X: X_test, Y: Y_test})
    test_writer.add_summary(test_summary, i)

    if i % 100 == 0:
        print("Iteration %i    train loss: %.10f  " % (i, model_loss), end="")
        print(np.linalg.norm(debug - y_mean))

writer.close()
test_writer.close()
