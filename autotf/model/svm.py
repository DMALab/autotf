from __future__ import division, print_function, absolute_import

from autotf.model.base_model import BaseModel
import tensorflow as tf
import numpy as np
import random

class SVM(BaseModel):

    default_param = {
        "metrics" : [],
        "optimizer" : "sgd",
        "learning_rate" : 1e-2,
        "batch_size" : 256,
        "num_epochs" : 10
    }

    def __init__(self, feature_num, param=None):
        self.feature_num = feature_num
        self.sess = tf.Session()
        if param:
            self.set_parameter(param)

    def build_model(self):
        with tf.variable_scope("linear_regression"):
            self.input_features = tf.placeholder(tf.float32, [None, self.feature_num], name="input_features")
            self.ground_truth = tf.placeholder(tf.float32, [None], name="ground_truth")
            self.weight = tf.get_variable("weight", [self.feature_num], tf.float32,
                                          tf.truncated_normal_initializer())
            self.bias = tf.get_variable("bias", [], tf.float32, tf.truncated_normal_initializer())
            self.output = tf.matmul(self.input_features, self.weight) + self.bias
            self.loss = tf.losses.hinge_loss(self.ground_truth, self.output)

    def set_parameter(self, param):
        for name, default_value in self.default_param:
            param.set_default(name, default_value)

        self.build_model()

        metrics = [self.get_metric(metric) for metric in param["metrics"]]
        self.metrics = [metric_fun(self.output, self.ground_truth) for metric_fun in metrics]

        optimizer = param["optimizer"]
        learning_rate = param["learning_rate"]
        self.optimizer = optimizer(learning_rate).minimize(self.loss)

        self.batch_size = param["batch_size"]
        self.num_epochs = param["num_epochs"]

    def get_batch(self, feed_data):
        X = np.ndarray([self.batch_size, self.feature_num], np.float32)
        Y = np.ndarray([self.batch_size], np.float32)

        input_features = feed_data["input_features"]
        ground_truth = feed_data["ground_truth"]

        total_data = np.shape(input_features)[0]
        batch_num = total_data // self.batch_size

        for _ in range(batch_num):
            for i in range(self.batch_size):
                r = random.randint(total_data)
                X[i] = input_features[r]
                Y[i] = ground_truth[r]
            yield { "input_features" : X, "ground_truth" : Y }

    def train(self, feed_data):
        if "input_features" not in feed_data or "ground_truth" not in feed_data:
            return -1
        self.sess.run(tf.global_variables_initializer())
        for epoch in range(self.num_epochs):
            for batch in self.get_batch(feed_data):
                feed_dict = {
                    self.input_features : batch["input_features"],
                    self.ground_truth : batch["ground_truth"]
                }
                _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)

    def evaluate(self, feed_data):
        input_features = feed_data["input_features"]
        ground_truth = feed_data["ground_truth"]
        output = []
        loss = []

        total_data = np.shape(input_features)[0]
        for batch_start in range(0, total_data, self.batch_size):
            input_batch = input_features[batch_start : batch_start + self.batch_size]
            feed_dict = {
                self.input_features : input_batch,
            }
            result = self.sess.run([self.output, self.loss], feed_dict)
            output.append(result[0])
            loss = result[1]

        output = np.concatenate(output, axis=0)
        loss = np.mean(loss)
        return loss, [metric(output, ground_truth) for metric in self.metrics]

    def predict(self, feed_data):
        input_features = feed_data["input_features"]
        feed_dict = {
            self.input_features : input_features,
        }
        return self.sess.run(self.output, feed_dict)

    def model_save(self, path):
        return

    def model_load(self, path):
        return
