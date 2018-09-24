from __future__ import division, print_function, absolute_import

import tensorflow as tf

class BaseModel(object):
    """A Model is an implementation of a machine learning algorithm with TensorFlow
    """

    loss_dict = {
        "square_loss" :
            lambda output, label:
                tf.reduce_mean(tf.square(output - label)),
        "cross_entropy" :
            lambda output, label:
                tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=output),
        "cross_entropy_1d" :
            lambda output, label:
                tf.nn.sigmoid_cross_entropy_with_logits(labels=label, logits=output),
    }

    optimizer_dict = {
        "sgd" : tf.train.GradientDescentOptimizer,
    }

    metric_dict = {

    }

    def __init__(self):
        return

    def set_parameter(self, param):
        return

    def train(self, data):
        return

    def predict(self, data):
        return

    def evaluate(self, data):
        return

    def model_load(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        return


    def model_save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)
        return


    @classmethod
    def get_loss(cls, loss_name):
        cls.loss_dict.get(loss_name)

    @classmethod
    def get_optimizer(cls, optimizer_name):
        cls.optimizer_dict.get(optimizer_name)

    @classmethod
    def get_metric(cls, metric_name):
        cls.metric_dict.get(metric_name)
