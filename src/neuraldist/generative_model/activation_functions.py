"""
    Definition of special activation functions
"""

import tensorflow as tf

def softplus(beta):
    # tf.nn.softplus is numerically stable (no overflow of exp for large beta * x)
    func = lambda x: tf.nn.softplus(beta * x) / beta
    func.__name__ = 'softplus_' + str(beta)
    return func