# -*- coding=utf-8 -*-
from __future__ import division, print_function, absolute_import

from base_model import BaseModel
from util import *

import tensorflow as tf
import pickle


class GoogleNetV3(BaseModel):
    default_param = {
        "loss": "square_loss",
        "metrics": ["loss"],
        "optimizer": "sgd",
        "learning_rate": 1e-2,
        "batch_size": 100,
        "num_epochs": 25,
    }

    def __init__(self, classnum):
        self.class_num = classnum
        self.sess = tf.Session()
        self.summary = []
        self.scope = {}

    def build_model(self):

        # 训练数据
        self.inputs = tf.placeholder(tf.float32, shape=[None, 299, 299, 3])
        #299*299*3

        # 训练标签数据
        self.labels = tf.placeholder(tf.float32, shape=[None, self.class_num])


        self.conv2d_1= self.conv2d('conv2d_1', self.inputs, 32, 3, 2,padding="VALID")
        # 149*149*32

        self.conv2d_2 = self.conv2d('conv2d_2', self.conv2d_1,32, 3, 1,padding="VALID")
        # 147*147*32

        self.conv2d_3 = self.conv2d('conv2d_3', self.conv2d_2, 64, 3, 1,padding="SAME")
        #147*147*64

        self.pool_1 = self.max_pool("pool_1",self.conv2d_3,3,2,padding="VALID")
        # 73*73*64

        self.conv2d_4= self.conv2d('conv2d_4', self.pool_1, 80, 3, 1,padding="VALID")
        # 71*71*80

        self.conv2d_5 = self.conv2d('conv2d_5', self.conv2d_4,192, 3, 1,padding="SAME")
        # 71*71*192

        self.pool_2 = self.max_pool('pool_2', self.conv2d_5,3,2,padding="VALID")
        #35*35*192




        #35*35*192
        self.inception_3a_output = self.InceptionV3_1("inception_3a", self.pool_2, 64, 48, 64, 64, 96, 32)
        # 35*35*256

        self.inception_3b_output = self.InceptionV3_1("inception_3b", self.inception_3a_output, 64, 48, 64, 64, 96, 64)

        # 64 + 64 + 96 + 64 = 288
        # 35*35*288

        self.inception_3c_output = self.InceptionV3_1("inception_3c", self.inception_3b_output, 64, 48, 64, 64, 96, 64)

        # 64 + 64 + 96 + 64 = 288
        # 35*35*288

        self.inception_4a_output = self.InceptionV3_2("inception_4a", self.inception_3c_output, 384, 64, 96)
        #17*17*768


        self.inception_4b_output = self.InceptionV3_3("inception_4b", self.inception_4a_output, 192, 128, 192, 128, 192,192)
        #17*17*768

        self.inception_4c_output = self.InceptionV3_3("inception_4c", self.inception_4b_output, 192, 160, 192, 160, 192, 192)
        #17*17*768

        self.inception_4d_output = self.InceptionV3_3("inception_4d", self.inception_4c_output, 192, 160, 192, 160, 192,192)
        #17*17*768

        self.inception_4e_output = self.InceptionV3_3("inception_4e", self.inception_4d_output, 192, 192, 192, 192, 192, 192)
        #17*17*768

        self.inception_5a_output = self.InceptionV3_4("inception_5a", self.inception_4e_output, 192, 320, 192)
        # 8*8*1280

        self.inception_5b_output = self.InceptionV3_5("inception_5b", self.inception_5a_output, 320, 384, 448, 384, 192)
        # 8*8*2048

        self.inception_5c_output = self.InceptionV3_5("inception_5c", self.inception_5b_output, 320, 384, 448, 384, 192)
        # 8*8*2048
        cur = self.inception_5c_output.get_shape()
        print(cur)

        self.pool_final = self.avg_pool('pool_final', self.inception_5c_output, 8, 1,padding="VALID")
        # 1x1x2048


        self.pred = self.conv2d_clean('loss3_classifier', self.pool_final, self.class_num,1)
        # class number
        self.pred = self.pred[:]
        self.pred = tf.reduce_sum(self.pred,axis=1)
        self.pred = tf.reduce_sum(self.pred, axis=1)
        cur = self.pred.get_shape()
        print(cur)


    def InceptionV3_1(self,layer_name,inputs,size1x1,size3x3reduce,size3x3,size5x5reduce,size5x5,sizepool):
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope
            i1x1 = self.conv2d('1x1', inputs, size1x1, 1, 1)

            i3x3_reduce = self.conv2d('3x3_reduce', inputs, size3x3reduce, 1, 1)
            i3x3 = self.conv2d('3x3', i3x3_reduce, size3x3, 3, 1)

            i5x5_reduce = self.conv2d('5x5_reduce', inputs, size5x5reduce, 1, 1)
            i5x5 = self.conv2d('5x5', i5x5_reduce, size5x5, 5, 1)

            pool = self.max_pool('pool', inputs, 3, 1)
            pool_proj = self.conv2d('pool_proj', pool, sizepool, 1, 1)
            output = self.concat('output',[i1x1,i3x3, i5x5,pool_proj])
            return output
    def InceptionV3_2(self,layer_name,inputs,depth0,depth1_1,depth1_2):
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope

            i1x1 = self.conv2d('1x1', inputs, depth0, 3, 2,padding="VALID")

            i3x3_reduce = self.conv2d('3x3_reduce', inputs, depth1_1, 1, 1)
            i3x3_1 = self.conv2d('3x3_1', i3x3_reduce, depth1_2, 3, 1,padding="SAME")
            i3x3_2 = self.conv2d('3x3_2', i3x3_1, depth1_2, 3, 2,padding="VALID")

            pool = self.max_pool('pool', inputs, 3, 2,padding="VALID")
            with tf.name_scope("concat"):
                output = tf.concat([i1x1, i3x3_2, pool], axis=3)
                return output

    def InceptionV3_3(self, layer_name, inputs, depth0,depth1_1,depth1_2,depth2_1,depth2_2,depth3):
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope

            i1x1 = self.conv2d('1x1', inputs, depth0, 3, 1)

            i1x1_reduce_1 = self.conv2d('i1x1_reduce_1', inputs, depth1_1, 1, 1)
            i1x7_1 = self.conv2d_lr('i1x7_1', i1x1_reduce_1, depth1_1, 1,7, 1)
            i7x1_1 = self.conv2d_lr('i7x1_1', i1x7_1, depth1_2, 7,1, 1)

            i1x1_reduce_2 =  self.conv2d('i1x1_reduce_2', inputs, depth2_1, 1, 1)
            i1x7_a = self.conv2d_lr('i1x7_a', i1x1_reduce_2, depth2_1, 7,1, 1)
            i7x1_b = self.conv2d_lr('i7x1_b', i1x7_a, depth2_1, 1,7, 1)
            i1x7_c = self.conv2d_lr('i1x7_c', i7x1_b, depth2_1, 7,1, 1)
            i7x1_d = self.conv2d_lr('i7x1_d', i1x7_c, depth2_2, 1,7, 1)

            pool = self.avg_pool('pool', inputs, 3, 1)
            pool_proj = self.conv2d("pool_proj",pool,depth3,1,1)

            with tf.name_scope("concat"):
                output = tf.concat([i1x1, i7x1_1, i7x1_d,pool_proj], axis=3)
                return output


    def InceptionV3_4(self, layer_name, inputs, depth1_1, depth1_2, depth2):

        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope
            i1x1 = self.conv2d("1x1",inputs,depth1_1,1,1)
            i1x1 = self.conv2d('3x3', i1x1, depth1_2, 3, 2,padding="VALID")


            i1x1_reduce_1 = self.conv2d('i1x1_reduce_1', inputs, depth2, 1, 1,padding="SAME")
            i1x7_1 = self.conv2d_lr('i1x7_1', i1x1_reduce_1, depth2, 1,7, 1,padding="SAME")
            i7x1_2 = self.conv2d_lr('i7x1_2', i1x7_1, depth2, 7,1, 1,padding="SAME")
            i7x1_3 = self.conv2d('i7x1_3', i7x1_2, depth2, 3, 2,padding="VALID")


            pool = self.max_pool('pool', inputs, 3, 2,padding="VALID")

            with tf.name_scope("concat"):
                output = tf.concat([i1x1, i7x1_3,pool], axis=3)
                return output
    def InceptionV3_5(self, layer_name, inputs, depth0, depth1, depth2_1,depth2_2,depth3):

        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope
            i1x1 = self.conv2d("1x1",inputs,depth0,1,1)

            i1x1_reduce_1 = self.conv2d('i1x1_reduce_1', inputs, depth1, 1, 1)
            i1x3_1_1 = self.conv2d_lr('i1x3_1_1', i1x1_reduce_1, depth1, 1,3, 1)
            i3x1_2_1 = self.conv2d_lr('i3x1_2_1', i1x1_reduce_1, depth1, 3,1, 1)
            with tf.name_scope("concat_1"):
                concat_1 = tf.concat([i1x3_1_1,i3x1_2_1],axis=3)


            i1x1_reduce_2 = self.conv2d('i1x1_reduce_2_a', inputs, depth2_1, 1, 1)
            i3x3_reduce_2 = self.conv2d('i3x3_reduce_2_b', i1x1_reduce_2, depth2_2, 3, 1)
            i1x3_1_2 = self.conv2d_lr('i1x3_1_2', i3x3_reduce_2, depth2_2, 1,3, 1)
            i3x1_2_2 = self.conv2d_lr('i3x1_2_2', i3x3_reduce_2, depth2_2, 3,1, 1)
            with tf.name_scope("concat_2"):
                concat_2 = tf.concat([i1x3_1_2,i3x1_2_2],axis=3)


            pool = self.avg_pool('pool', inputs, 3,1)
            pool_proj = self.conv2d("pool_proj",pool,depth3,1,1)

            with tf.name_scope("concat"):
                output = tf.concat([i1x1,concat_1, concat_2,pool_proj], axis=3)
                return output




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
        self.sess.run(tf.global_variables_initializer())
        print(self.num_epochs)

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
    def conv2d_clean(self, layer_name, inputs, out_channels, kernel_size, strides=1, padding='SAME'):
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
            return inputs
    def conv2d_bn(self, layer_name, inputs, out_channels, kernel_size, strides=1, padding='SAME'):
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
            inputs = tf.layers.batch_normalization(inputs)
            inputs = tf.nn.relu(inputs, name='relu')
            return inputs
    def conv2d_lr(self, layer_name, inputs, out_channels, kernel_size_l,kernel_size_r, strides=1, padding='SAME'):
        in_channels = inputs.get_shape()[-1]
        with tf.variable_scope(layer_name) as scope:
            self.scope[layer_name] = scope
            w = tf.get_variable(name='weights',
                                trainable=True,
                                shape=[kernel_size_l, kernel_size_r, in_channels, out_channels],
                                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.get_variable(name='biases',
                                trainable=True,
                                shape=[out_channels],
                                initializer=tf.constant_initializer(0.0))
            inputs = tf.nn.conv2d(inputs, w, [1, strides, strides, 1], padding=padding, name='conv')
            inputs = tf.nn.bias_add(inputs, b, name='bias_add')
            inputs = tf.nn.relu(inputs, name='relu')
            return inputs

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

    def lrn(self, layer_name, inputs, depth_radius=5, alpha=0.0001, beta=0.75):
        with tf.name_scope(layer_name):
            return tf.nn.local_response_normalization(name='pool1_norm1', input=inputs, depth_radius=depth_radius,
                                                      alpha=alpha, beta=beta)

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

    def model_load(self, path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        return

    def model_save(self, path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)
        return

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
