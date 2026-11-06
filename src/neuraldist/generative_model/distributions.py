import tensorflow as tf
import numpy as np

class StandardNormal:
    def __init__(self, dim):
        self.dim = dim
        self.log_norm = -0.5 * dim * tf.math.log(2 * tf.constant(np.pi, dtype=tf.float32))

    def sample(self, n):
        return tf.random.normal(shape=(n, self.dim))

    def log_prob(self, x):
        return self.log_norm - 0.5 * tf.reduce_sum(tf.square(x), axis=-1)
