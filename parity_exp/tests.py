from graph import X, Y, train_step
import numpy as np
import tensorflow as tf

def test_variables_changing():

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    before = sess.run(tf.trainable_variables())
    _ = sess.run(
        train_step,
        feed_dict={
            X: np.round(np.random.rand(100, 8)),
            Y: np.round(np.random.rand(100)),
        }
    )
    after = sess.run(tf.trainable_variables())
    for b, a in zip(before, after):
        assert (b != a).any()
    print("Success")

if __name__ == "__main__":
    test_variables_changing()
