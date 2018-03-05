import numpy as np
import tensorflow as tf
# import itertools
# import math

from generate_bits import bits_prepared
from graph_mtl import X, Y1, Y2, Y3, Y4, train_step
from graph_mtl import merged, loss, correct1
# from graph import X, Y, train_step
# from graph import merged, loss, correct1
# from batcher import Batcher


# Constants
max_iter = 1000000
batch_size = 32
print_interval = 20


# Data loading
X_full, Y1_full, Y2_full, Y3_full, Y4_full = bits_prepared()
# X_full = np.array(X_full, dtype=float)
# Y1_full = np.array(Y1_full, dtype=float)

random_order = np.random.permutation(X_full.shape[0])


def rand_and_split(arr):
    return np.split(np.take(arr, random_order, axis=0, out=arr), 2)


X_train,   X_test = rand_and_split(X_full)
Y1_train, Y1_test = rand_and_split(Y1_full)
Y2_train, Y2_test = rand_and_split(Y2_full)
Y3_train, Y3_test = rand_and_split(Y3_full)
Y4_train, Y4_test = rand_and_split(Y4_full)


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

writer = tf.summary.FileWriter("/tmp/log/parity/train", sess.graph)

test_writer = tf.summary.FileWriter("/tmp/log/parity/test", sess.graph)

for i in range(max_iter):
    summary, model_loss, _, debug = sess.run(
        [merged, loss, train_step, correct1],
        {
            X: X_train,
            Y1: Y1_train,
            Y2: Y2_train,
            Y3: Y3_train,
            Y4: Y4_train,
        })
    writer.add_summary(summary, i)

    test_summary = sess.run(
        merged,
        {
            X: X_test,
            Y1: Y1_test,
            Y2: Y2_test,
            Y3: Y3_test,
            Y4: Y4_test,
        })
    test_writer.add_summary(test_summary, i)

    if i % print_interval == 0:
        print("Iteration %i    train loss: %.3E  " % (i, model_loss))

writer.close()
test_writer.close()
