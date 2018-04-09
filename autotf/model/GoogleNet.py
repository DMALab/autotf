#-*- coding=utf-8 -*-
from __future__ import division, print_function, absolute_import

from base_model import BaseModel
import util

import tensorflow as tf
import tflearn

X, Y = oxflower17.load_data("data",one_hot=True, resize_pics=(227, 227))

class GoogleNet(BaseModel):
    default_param = {
        "loss" : "square_loss",
        "metrics" : ["loss"],
        "optimizer" : "sgd",
        "learning_rate" : 1e-2,
        "batch_size" : 100,
        "num_epochs" : 25,
        "keep_prob":0.75
    }

    def __init__(self, feature_num,classnum):
        self.feature_num = feature_num
        self.class_num = classnum
        self.sess = tf.Session()

    def build_model(self):

        # 训练数据
        self.inputs = tf.placeholder(tf.float32, shape=[None, 227, 227, 3])
        # 训练标签数据
        self.labels = tf.placeholder(tf.float32, shape=[None, 17])
        # dropout
        self.keep_prob = tf.placeholder(tf.float32)

        self.conv1_7x7_s2 = conv2d('conv1_7x7_s2', self.inputs, 64, 7, 2)
        self.pool1_3x3_s2 = max_pool('pool1_3x3_s2', self.conv1_7x7_s2, 3, 2)
        self.pool1_norm1 = lrn('pool1_norm1', self.pool1_3x3_s2)
        self.conv2_3x3_reduce = conv2d('conv2_3x3_reduce', self.pool1_norm1, 64, 1, 1)
        self.conv2_3x3 = conv2d('conv2_3x3', self.conv2_3x3_reduce, 192, 3, 1)
        self.conv2_norm2 = lrn('conv2_norm2', self.conv2_3x3)
        self.pool2_3x3_s2 = max_pool('pool2_3x3_s2', self.conv2_norm2, 3, 2)

        self.inception_3a_1x1 = conv2d('inception_3a_1x1', self.pool2_3x3_s2, 64, 1, 1)
        self.inception_3a_3x3_reduce = conv2d('inception_3a_3x3_reduce', self.pool2_3x3_s2, 96, 1, 1)
        self.inception_3a_3x3 = conv2d('inception_3a_3x3', self.inception_3a_3x3_reduce, 128, 3, 1)
        self.inception_3a_5x5_reduce = conv2d('inception_3a_5x5_reduce', self.pool2_3x3_s2, 16, 1, 1)
        self.inception_3a_5x5 = conv2d('inception_3a_5x5', self.inception_3a_5x5_reduce, 32, 5, 1)
        self.inception_3a_pool = max_pool('inception_3a_pool', self.pool2_3x3_s2, 3, 1)
        self.inception_3a_pool_proj = conv2d('inception_3a_pool_proj', self.inception_3a_pool, 32, 1, 1)
        self.inception_3a_output = concat('inception_3a_output', [self.inception_3a_1x1, self.inception_3a_3x3, self.inception_3a_5x5,
                                                                  self.inception_3a_pool_proj])

        self.inception_3b_1x1 = conv2d('inception_3b_1x1', self.inception_3a_output, 128, 1, 1)
        self.inception_3b_3x3_reduce = conv2d('inception_3b_3x3_reduce', self.inception_3a_output, 128, 1, 1)
        self.inception_3b_3x3 = conv2d('inception_3b_3x3', self.inception_3b_3x3_reduce, 192, 3, 1)
        self.inception_3b_5x5_reduce = conv2d('inception_3b_5x5_reduce', self.inception_3a_output, 32, 1, 1)
        self.inception_3b_5x5 = conv2d('inception_3b_5x5', self.inception_3b_5x5_reduce, 96, 5, 1)
        self.inception_3b_pool = max_pool('inception_3b_pool', self.inception_3a_output, 3, 1)
        self.inception_3b_pool_proj = conv2d('inception_3b_pool_proj', self.inception_3b_pool, 64, 1, 1)
        self.inception_3b_output = concat('inception_3b_output', [self.inception_3b_1x1, self.inception_3b_3x3, self.inception_3b_5x5,
                                                                       self.inception_3b_pool_proj])

        self.pool3_3x3_s2 = max_pool('pool3_3x3_s2', self.inception_3b_output, 3, 2)
        self.inception_4a_1x1 = conv2d('inception_4a_1x1', self.pool3_3x3_s2, 192, 1, 1)
        self.inception_4a_3x3_reduce = conv2d('inception_4a_3x3_reduce', self.pool3_3x3_s2, 96, 1, 1)
        self.inception_4a_3x3 = conv2d('inception_4a_3x3', self.inception_4a_3x3_reduce, 208, 3, 1)
        self.inception_4a_5x5_reduce = conv2d('inception_4a_5x5_reduce', self.pool3_3x3_s2, 16, 1, 1)
        self.inception_4a_5x5 = conv2d('inception_4a_5x5', self.inception_4a_5x5_reduce, 48, 5, 1)
        self.inception_4a_pool = max_pool('inception_4a_pool', self.pool3_3x3_s2, 3, 1)
        self.inception_4a_pool_proj = conv2d('inception_4a_pool_proj', self.inception_4a_pool, 64, 1, 1)
        self.inception_4a_output = concat('inception_4a_output', [self.inception_4a_1x1, self.inception_4a_3x3, self.inception_4a_5x5,
                                                                       self.inception_4a_pool_proj])

        self.inception_4b_1x1 = conv2d('inception_4b_1x1', self.inception_4a_output, 160, 1, 1)
        self.inception_4b_3x3_reduce = conv2d('inception_4b_3x3_reduce', self.inception_4a_output, 112, 1, 1)
        self.inception_4b_3x3 = conv2d('inception_4b_3x3', self.inception_4b_3x3_reduce, 224, 3, 1)
        self.inception_4b_5x5_reduce = conv2d('inception_4b_5x5_reduce', self.inception_4a_output, 24, 1, 1)
        self.inception_4b_5x5 = conv2d('inception_4b_5x5', self.inception_4b_5x5_reduce, 64, 5, 1)
        self.inception_4b_pool = max_pool('inception_4b_pool', self.inception_4a_output, 3, 1)
        self.inception_4b_pool_proj = conv2d('inception_4b_pool_proj', self.inception_4b_pool, 64, 1, 1)
        self.inception_4b_output = concat('inception_4b_output', [self.inception_4b_1x1, self.inception_4b_3x3, self.inception_4b_5x5,
                                                                       self.inception_4b_pool_proj])

        self.inception_4c_1x1 = conv2d('inception_4c_1x1', self.inception_4b_output, 128, 1, 1)
        self.inception_4c_3x3_reduce = conv2d('inception_4c_3x3_reduce', self.inception_4b_output, 128, 1, 1)
        self.inception_4c_3x3 = conv2d('inception_4c_3x3', self.inception_4c_3x3_reduce, 256, 3, 1)
        self.inception_4c_5x5_reduce = conv2d('inception_4c_5x5_reduce', self.inception_4b_output, 24, 1, 1)
        self.inception_4c_5x5 = conv2d('inception_4c_5x5', self.inception_4c_5x5_reduce, 64, 5, 1)
        self.inception_4c_pool = max_pool('inception_4c_pool', self.inception_4b_output, 3, 1)
        self.inception_4c_pool_proj = conv2d('inception_4c_pool_proj', self.inception_4c_pool, 64, 1, 1)
        self.inception_4c_output = concat('inception_4c_output', [self.inception_4c_1x1, self.inception_4c_3x3, self.inception_4c_5x5,
                                                                  self.inception_4c_pool_proj])

        self.inception_4d_1x1 = conv2d('inception_4d_1x1', inception_4c_output, 112, 1, 1)
        self.inception_4d_3x3_reduce = conv2d('inception_4d_3x3_reduce', inception_4c_output, 144, 1, 1)
        self.inception_4d_3x3 = conv2d('inception_4d_3x3', inception_4d_3x3_reduce, 288, 3, 1)
        self.inception_4d_5x5_reduce = conv2d('inception_4d_5x5_reduce', inception_4c_output, 32, 1, 1)
        self.inception_4d_5x5 = conv2d('inception_4d_5x5', self.inception_4d_5x5_reduce, 64, 5, 1)
        self.inception_4d_pool = max_pool('inception_4d_pool', self.inception_4c_output, 3, 1)
        self.inception_4d_pool_proj =conv2d('inception_4d_pool_proj', self.inception_4d_pool, 64, 1, 1)
        self.inception_4d_output = concat('inception_4d_output', [self.inception_4d_1x1, self.inception_4d_3x3, self.inception_4d_5x5,
                                                                       self.inception_4d_pool_proj])

        self.inception_4e_1x1 = self.conv2d('inception_4e_1x1', self.inception_4d_output, 256, 1, 1)
        self.inception_4e_3x3_reduce = self.conv2d('inception_4e_3x3_reduce', self.inception_4d_output, 160, 1, 1)
        self.inception_4e_3x3 = self.conv2d('inception_4e_3x3', self.inception_4e_3x3_reduce, 320, 3, 1)
        self.inception_4e_5x5_reduce = self.conv2d('inception_4e_5x5_reduce', self.inception_4d_output, 32, 1, 1)
        self.inception_4e_5x5 = self.conv2d('inception_4e_5x5', self.inception_4e_5x5_reduce, 128, 5, 1)
        self.inception_4e_pool = self.max_pool('inception_4e_pool', self.inception_4d_output, 3, 1)
        self.inception_4e_pool_proj = self.conv2d('inception_4e_pool_proj', self.inception_4e_pool, 128, 1, 1)
        self.inception_4e_output = self.concat('inception_4e_output', [self.inception_4e_1x1, self.inception_4e_3x3, self.inception_4e_5x5,
                                                                       self.inception_4e_pool_proj])

        self.pool4_3x3_s2 = self.max_pool('pool4_3x3_s2', self.inception_4e_output, 3, 2)
        self.inception_5a_1x1 = self.conv2d('inception_5a_1x1', self.pool4_3x3_s2, 256, 1, 1)
        self.inception_5a_3x3_reduce = self.conv2d('inception_5a_3x3_reduce', self.pool4_3x3_s2, 160, 1, 1)
        self.inception_5a_3x3 = self.conv2d('inception_5a_3x3', self.inception_5a_3x3_reduce, 320, 3, 1)
        self.inception_5a_5x5_reduce = self.conv2d('inception_5a_5x5_reduce', self.pool4_3x3_s2, 32, 1, 1)
        self.inception_5a_5x5 = self.conv2d('inception_5a_5x5', self.inception_5a_5x5_reduce, 128, 5, 1)
        self.inception_5a_pool = self.max_pool('inception_5a_pool', self.pool4_3x3_s2, 3, 1)
        self.inception_5a_pool_proj = self.conv2d('inception_5a_pool_proj', self.inception_5a_pool, 128, 1, 1)
        self.inception_5a_output = self.concat('inception_5a_output', [self.inception_5a_1x1, self.inception_5a_3x3, self.inception_5a_5x5,
                                                                       self.inception_5a_pool_proj])

        self.inception_5b_1x1 = self.conv2d('inception_5b_1x1', self.inception_5a_output, 384, 1, 1)
        self.inception_5b_3x3_reduce = self.conv2d('inception_5b_3x3_reduce', self.inception_5a_output, 192, 1, 1)
        self.inception_5b_3x3 = self.conv2d('inception_5b_3x3', self.inception_5b_3x3_reduce, 384, 3, 1)
        self.inception_5b_5x5_reduce = self.conv2d('inception_5b_5x5_reduce', self.inception_5a_output, 48, 1, 1)
        self.inception_5b_5x5 = self.conv2d('inception_5b_5x5', self.inception_5b_5x5_reduce, 128, 5, 1)
        self.inception_5b_pool = self.max_pool('inception_5b_pool', self.inception_5a_output, 3, 1)
        self.inception_5b_pool_proj = self.conv2d('inception_5b_pool_proj', self.inception_5b_pool, 128, 1, 1)
        self.inception_5b_output = self.concat('inception_5b_output', [self.inception_5b_1x1, self.inception_5b_3x3, self.inception_5b_5x5,
                                                                       self.inception_5b_pool_proj])

        self.pool5_7x7_s1 = self.avg_pool('pool5_7x7_s1', self.inception_5b_output, 7, 1)
        self.pool5_drop_7x7_s1 = self.dropout('pool5_drop_7x7_s1', self.pool5_7x7_s1, 0.6)

        self.logits = self.fc('loss3_classifier', self.pool5_drop_7x7_s1, out_nodes=10)


    def set_parameter(self, param):
        for name in self.default_param:
            if name not in param:
                param[name] = self.default_param[name]

        self.build_model()
        # 定义交叉熵损失函数
        self.keep_prob_value = param["keep_prob"]

        loss_fun = param["loss"]
        self.loss = loss_fun(self.output, self.ground_truth)

        metrics = [self.get_metric(metric) for metric in param["metrics"]]
        self.metrics = [metric_fun(self.output, self.ground_truth) for metric_fun in metrics]

        optimizer = param["optimizer"]
        learning_rate = param["learning_rate"]
        self.optimizer = optimizer(learning_rate).minimize(self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

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
            i = int(0)
            for batch in self.get_batch(feed_data):
                feed_dict = {
                    self.inputs : batch["batch_xs"],
                    self.labels : batch["batch_ys"],
                    self.keep_prob: self.keep_prob_value,
                }
                _, loss, train_accuracy = self.sess.run([self.train_op, self.cost,self.accuracy], feed_dict=feed_dict)
                i = i + 1
                if (i%100 == 0):
                    print("step %d, training accuracy %g" % (i, train_accuracy))
                avg_cost +=  loss
            avg_cost /= int(feed_data.train.num_examples/self.batch_size)
            print(avg_cost)

    def evaluate(self, feed_data):
        return