#-*- coding=utf-8 -*-
from __future__ import division, print_function, absolute_import

from autotf.model.base_model import BaseModel
from autotf.model.helper import *
import tensorflow as tf

class Vgg19(BaseModel):

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
        self.model = None
        self.sess = tf.Session()

    def build_model(self):
        VGG_MEAN = [103.939, 116.779, 123.68]
        with tf.variable_scope("VGG"):
            # 训练数据
            self.inputs = tf.placeholder(tf.float32, shape=[None, 224, 224, 3])
            # 训练标签数据
            self.labels = tf.placeholder(tf.float32, shape=[None, self.class_num])
            # dropout
            self.keep_prob = tf.placeholder(tf.float32)

            # Convert RGB to BGR
            red, green, blue = tf.split(axis=3, num_or_size_splits=3, value=self.inputs)
            bgr = tf.concat(axis=3, values=[
                blue * 255.0 - VGG_MEAN[0],
                green * 255.0 - VGG_MEAN[1],
                red * 255.0 - VGG_MEAN[2],
            ])

            self.conv1_1 = conv_layer(bgr, 3, 64, "conv1_1")
            self.conv1_2 = conv_layer(self.conv1_1, 64, 64, "conv1_2")
            self.pool1 = max_pool(self.conv1_2, 'pool1')

            self.conv2_1 = conv_layer(self.pool1, 64, 128, "conv2_1")
            self.conv2_2 = conv_layer(self.conv2_1, 128, 128, "conv2_2")
            self.pool2 = max_pool(self.conv2_2, 'pool2')

            self.conv3_1 = conv_layer(self.pool2, 128, 256, "conv3_1")
            self.conv3_2 = conv_layer(self.conv3_1, 256, 256, "conv3_2")
            self.conv3_3 = conv_layer(self.conv3_2, 256, 256, "conv3_3")
            self.conv3_4 = conv_layer(self.conv3_3, 256, 256, "conv3_4")
            self.pool3 = max_pool(self.conv3_4, 'pool3')

            self.conv4_1 = conv_layer(self.pool3, 256, 512, "conv4_1")
            self.conv4_2 = conv_layer(self.conv4_1, 512, 512, "conv4_2")
            self.conv4_3 = conv_layer(self.conv4_2, 512, 512, "conv4_3")
            self.conv4_4 = conv_layer(self.conv4_3, 512, 512, "conv4_4")
            self.pool4 = max_pool(self.conv4_4, 'pool4')

            self.conv5_1 = conv_layer(self.pool4, 512, 512, "conv5_1")
            self.conv5_2 = conv_layer(self.conv5_1, 512, 512, "conv5_2")
            self.conv5_3 = conv_layer(self.conv5_2, 512, 512, "conv5_3")
            self.conv5_4 = conv_layer(self.conv5_3, 512, 512, "conv5_4")
            self.pool5 = max_pool(self.conv5_4, 'pool5')

            self.fc6 = fc_layer(self.pool5, 25088, 4096, "fc6")  # 25088 = ((224 // (2 ** 5)) ** 2) * 512
            self.relu6 = tf.nn.relu(self.fc6)
            self.relu6 = tf.nn.dropout(self.relu6, self.keep_prob)

            self.fc7 = fc_layer(self.relu6, 4096, 4096, "fc7")
            self.relu7 = tf.nn.relu(self.fc7)
            self.relu7 = tf.nn.dropout(self.relu7, self.keep_prob)

            self.pred = fc_layer(self.relu7, 4096, self.class_num, "fc8")

    def set_parameter(self, param):
        for name in self.default_param:
            if name not in param:
                param[name] = self.default_param[name]

        self.build_model()
        # 定义交叉熵损失函数
        self.keep_prob_value = param["keep_prob"]

        loss_fun = self.get_loss(param["loss"])
        self.loss = loss_fun(self.output, self.ground_truth)

        metrics = [self.get_metric(metric) for metric in param["metrics"]]
        self.metrics = [metric_fun(self.output, self.ground_truth) for metric_fun in metrics]

        optimizer = self.get_optimizer(param["optimizer"])
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
