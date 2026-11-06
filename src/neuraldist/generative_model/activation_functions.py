"""
    Definition of special actifation functions
"""

import tensorflow as tf

def softplus(beta):
    func = lambda x: tf.math.log(1 + tf.exp(beta * x)) / beta
    func.__name__ = 'softplus_' + str(beta)
    return func