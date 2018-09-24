# -*- coding=utf-8 -*-
from __future__ import division, print_function, absolute_import

from base_model import BaseModel
from util import *

import tensorflow as tf
import pickle


class ResNet(BaseModel):
    default_param = {
        "loss": "square_loss",
        "metrics": ["loss"],
        "optimizer": "sgd",
        "learning_rate": 1e-2,
        "batch_size": 100,
        "num_epochs": 25,
        "block_num": 5,
    }

    def __init__(self, classnum):
        self.class_num = classnum
        self.sess = tf.Session()
        self.summary = []
        self.scope = {}

    def build_model(self):

        # 训练数据
        self.inputs = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
        #32*32*3

        # 训练标签数据
        self.labels = tf.placeholder(tf.float32, shape=[None, self.class_num])

        self.conv1= self.conv2d('conv2d', self.inputs, 16, 3, padding="SAME")
        # 149*149*32

        self.net = self.resnet_block("resnet1",self.conv1,block_num,16)
        self.net = self.resnet_block("resnet2",self.net,1,32,downsample=True)
        self.net = self.resnet_block("resnet3",self.net,self.block_num-1,32)
        self.net = self.resnet_block("resnet4", self.net, 1, 64,downsample=True)
        self.net = self.resnet_block("resnet5", self.net, self.block_num - 1, 64, downsample=True)
        # class number

        self.pred = self.fc("fully_connect",self.net,self.class_num)


    def resnet_block(self, layer_name, inputs, layer_number,out_channels, downsample=False,downsample_strides=2, activation="relu",
                     batch_norm=True, reuse=False):
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope
            resnet = inputs
            in_channels = inputs.get_shape()[-1]
            for i in range(layer_number):
                identity = resnet
                if not downsample:
                    downsample_strides = 1

                resnet = self.conv2d(str(i), resnet, out_channels, 3, downsample_strides, 'SAME')

                resnet = self.conv2d(str(i), resnet, out_channels, 3, downsample_strides, 'SAME')

                if downsample_strides >1:
                    identity = self.avg_pool(str(i)+"avg_pool", identity, downsample_strides, 1)
                    #NHWC do not change NHW so the pad [0,0][0,0][0,0]

                if in_channels != out_channels:
                    ch = (out_channels - in_channels)//2
                    identity = tf.pad(identity,[[0,0],[0,0],[0,0],[ch,ch]])
                    in_channels = out_channels

                resnet = resnet + identity
        return resnet


    def bottleneck_block(self, layer_name, inputs, layer_number,out_channels, downsample=False,downsample_strides=2, activation="relu",
                     batch_norm=True, reuse=False):
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope
            resnet = inputs
            in_channels = inputs.get_shape()[-1]
            for i in range(layer_number):
                identity = resnet
                if not downsample:
                    downsample_strides = 1

                resnet = self.conv2d(str(i), resnet, out_channels, 3, downsample_strides, 'SAME')

                resnet = self.conv2d(str(i), resnet, out_channels, 3, downsample_strides, 'SAME')

                if downsample_strides >1:
                    identity = self.avg_pool(str(i)+"avg_pool", identity, downsample_strides, 1)
                    #NHWC do not change NHW so the pad [0,0][0,0][0,0]

                if in_channels != out_channels:
                    ch = (out_channels - in_channels)//2
                    identity = tf.pad(identity,[[0,0],[0,0],[0,0],[ch,ch]])
                    in_channels = out_channels

                resnet = resnet + identity
        return resnet

    def set_parameter(self, param):
        for name in self.default_param:
            if name not in param:
                param[name] = self.default_param[name]

        self.build_model()

        # 定义交叉熵损失函数
        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.labels))

        optimizer = param["optimizer"]
        learning_rate = param["learning_rate"]
        self.optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.batch_size = param["batch_size"]
        self.num_epochs = param["num_epochs"]
        self.block_num = param["block_num"]
        self.sess.run(tf.global_variables_initializer())

    def get_batch(self, feed_data):
        X = feed_data["inputs"]
        Y = feed_data["labels"]
        totalbatch = int(len(X) / self.batch_size) + 1

        if ( len(X) / self.batch_size * self.batch_size == len(X)):
            totalbatch = totalbatch - 1

        for i in range(0, totalbatch):
            startindex = i * self.batch_size
            endindex = (i + 1) * self.batch_size
            batch_xs = X[startindex:endindex]
            batch_ys = Y[startindex:endindex]

            yield {"batch_xs": batch_xs, "batch_ys": batch_ys}
    def train_batch(self,feed_data):
        feed_dict = {
            self.inputs: feed_data["inputs"],
            self.labels: feed_data["labels"],
        }
        _, loss, train_accuracy = self.sess.run([self.optimizer, self.loss, self.accuracy], feed_dict=feed_dict)
        return train_accuracy

    def train(self, feed_data):

        trainstep = 0
        for epoch in range(0, self.num_epochs):
            avg_cost = 0.0

            totalaccuracy = 0.0
            for batch in self.get_batch(feed_data):
                feed_dict = {
                    self.inputs: batch["batch_xs"],
                    self.labels: batch["batch_ys"],
                }
                _, loss, train_accuracy = self.sess.run([self.optimizer, self.loss, self.accuracy], feed_dict=feed_dict)
                totalaccuracy += train_accuracy * len(batch["batch_xs"])
                avg_cost += loss
                trainstep = trainstep + 1
            totalaccuracy /= len(feed_data["inputs"])

            print("accuracy:" + "\t" + str(totalaccuracy) + "\t" + "loss:" + "\t" + str(avg_cost))


    def conv2d(self, layer_name, inputs, out_channels, kernel_size, strides=1, padding='SAME'):
        in_channels = inputs.get_shape()[-1]
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope
            w = tf.get_variable(name='weights',
                                trainable=True,
                                shape=[kernel_size, kernel_size, in_channels, out_channels],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='biases',
                                trainable=True,
                                shape=[out_channels],
                                initializer=tf.constant_initializer(0.0))
            inputs = tf.nn.conv2d(inputs, w, [1, strides, strides, 1], padding=padding, name='conv')
            inputs = tf.nn.bias_add(inputs, b, name='bias_add')
            inputs = tf.nn.relu(inputs, name='relu')
            return inputs


    def max_pool(self, layer_name, inputs, pool_size, strides, padding='SAME'):
        with tf.name_scope(layer_name):
            return tf.nn.max_pool(inputs, [1, pool_size, pool_size, 1], [1, strides, strides, 1], padding=padding,
                                  name=layer_name)


    def avg_pool(self, layer_name, inputs, pool_size, strides, padding='SAME'):
        with tf.name_scope(layer_name):
            return tf.nn.avg_pool(inputs, [1, pool_size, pool_size, 1], [1, strides, strides, 1], padding=padding,
                                  name=layer_name)


    def concat(self, layer_name, inputs):
        with tf.name_scope(layer_name):
            one_by_one = inputs[0]
            three_by_three = inputs[1]
            five_by_five = inputs[2]
            pooling = inputs[3]
            return tf.concat([one_by_one, three_by_three, five_by_five, pooling], axis=3)


    def dropout(self, layer_name, inputs, keep_prob):
        # dropout_rate = 1 - keep_prob
        with tf.name_scope(layer_name):
            return tf.nn.dropout(name=layer_name, x=inputs, keep_prob=keep_prob)


    def bn(self, layer_name, inputs, epsilon=1e-3):
        with tf.name_scope(layer_name):
            batch_mean, batch_var = tf.nn.moments(inputs, [0])
            inputs = tf.nn.batch_normalization(inputs, mean=batch_mean, variance=batch_var, offset=None,
                                               scale=None, variance_epsilon=epsilon)
            return inputs


    def fc(self, layer_name, inputs, out_nodes):
        shape = inputs.get_shape()
        if len(shape) == 4:  # x is 4D tensor
            size = shape[1].value * shape[2].value * shape[3].value
        else:  # x has already flattened
            size = shape[-1].value
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope
            w = tf.get_variable('weights',
                                shape=[size, out_nodes],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable('biases',
                                shape=[out_nodes],
                                initializer=tf.constant_initializer(0.0))
            flat_x = tf.reshape(inputs, [-1, size])
            inputs = tf.nn.bias_add(tf.matmul(flat_x, w), b)
            inputs = tf.nn.relu(inputs)
            return inputs


    def evaluate(self, feed_data):
        avg_loss = 0.0
        totalaccuracy = 0.0
        totallen = len(feed_data["inputs"])

        for batch in self.get_batch(feed_data):
            feed_dict = {
                self.inputs: batch["batch_xs"],
                self.labels: batch["batch_ys"]
            }
            loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            totalaccuracy += acc * len(batch["batch_xs"])
            avg_loss += loss
        avg_loss /= totallen
        totalaccuracy /= len(feed_data['inputs'])

        res = {"accuracy": totalaccuracy, "loss": avg_loss}
        return res


    def predict(self, feed_data):
        res = []
        for batch in self.get_batch(feed_data):
            feed_dict = {
                self.inputs: batch["batch_xs"]
            }

            pred = self.sess.run(self.pred, feed_dict=feed_dict)
            res.extend(pred.tolist())

        return res
