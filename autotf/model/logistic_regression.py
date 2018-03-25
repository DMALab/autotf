from __future__ import division, print_function, absolute_import

from base_model import BaseModel
import tensorflow as tf
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("./data/", one_hot=True)

class LogisticRegression(BaseModel):

    default_param = {
        "loss" : "square_loss",
        "metrics" : ["loss"],
        "optimizer" : "sgd",
        "learning_rate" : 1e-2,
        "batch_size" : 100,
        "num_epochs" : 25
    }

    def __init__(self, feature_num,classnum):
        self.feature_num = feature_num
        self.class_num = classnum
        self.model = None
        self.sess = tf.Session()

    def build_model(self):
        with tf.variable_scope("logistic_regression"):
            self.input_features = tf.placeholder(tf.float32, [None, self.feature_num], name="input_features")
            self.output_classes = tf.placeholder(tf.float32,[None,self.class_num],name='output_classes')
            self.weight = tf.Variable(tf.zeros([self.feature_num, self.class_num]))
            self.bias = tf.Variable(tf.zeros([self.class_num]))
            self.output = tf.nn.softmax(tf.matmul(self.input_features, self.weight) + self.bias)


    def set_parameter(self, param):
        for name  in self.default_param:
            if name not in param:
                param[name] = self.default_param[name]

        self.build_model()

        #loss_fun = param["loss"]
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.output_classes * tf.log(self.output), reduction_indices=1))
        #self.loss = loss_fun(self.output, self.ground_truth)

        #metrics = [self.get_metric(metric) for metric in param["metrics"]]
        #self.metrics = [metric_fun(self.output, self.ground_truth) for metric_fun in metrics]

        optimizer = param["optimizer"]
        learning_rate = param["learning_rate"]
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

        self.batch_size = param["batch_size"]
        self.num_epochs = param["num_epochs"]

    def get_batch(self, feed_data):
        for i in range(0,int(feed_data.train.num_examples/self.batch_size)):
            batch_xs,batch_ys = feed_data.train.next_batch(self.batch_size)
            yield { "batch_xs" : batch_xs, "batch_ys" : batch_ys }

    def train(self, feed_data):
        self.sess.run(tf.global_variables_initializer())
        print("here")

        for epoch in range(self.num_epochs):
            avg_cost = 0.0

            for batch in self.get_batch(feed_data):
                feed_dict = {
                    self.input_features : batch["batch_xs"],
                    self.output_classes : batch["batch_ys"]
                }
                _, loss = self.sess.run([self.optimizer, self.loss], feed_dict=feed_dict)
                avg_cost +=  loss
            avg_cost /= int(feed_data.train.num_examples/self.batch_size)
            print(avg_cost)

    def evaluate(self, feed_data):

        return
params = {
    "loss" : "square_loss",
    "metrics" : ["loss"],
    "optimizer" : "sgd",
    "learning_rate" : 1e-2,
    "batch_size" : 100,
    "num_epochs" : 25
}

m = LogisticRegression(784,10)
m.set_parameter(params)
print(mnist.train.num_examples)
m.train(mnist)
