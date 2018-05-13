from __future__ import division, print_function, absolute_import

from base_model import BaseModel
import tensorflow as tf
import numpy as np
import random
from tensorflow.examples.tutorials.mnist import input_data

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
            self.inputs = tf.placeholder(tf.float32, [None, self.feature_num], name="inputs")
            self.labels = tf.placeholder(tf.float32,[None,self.class_num],name='labels')
            self.weight = tf.Variable(tf.zeros([self.feature_num, self.class_num]))
            self.bias = tf.Variable(tf.zeros([self.class_num]))
            self.pred = tf.nn.softmax(tf.matmul(self.inputs, self.weight) + self.bias)


    def set_parameter(self, param):
        for name  in self.default_param:
            if name not in param:
                param[name] = self.default_param[name]

        self.build_model()

        #loss_fun = param["loss"]
        self.loss = tf.reduce_mean(-tf.reduce_sum(self.labels * tf.log(self.pred), reduction_indices=1))
        #self.loss = loss_fun(self.output, self.ground_truth)

        #metrics = [self.get_metric(metric) for metric in param["metrics"]]
        #self.metrics = [metric_fun(self.output, self.ground_truth) for metric_fun in metrics]

        learning_rate = param["learning_rate"]
        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

        self.batch_size = param["batch_size"]
        self.num_epochs = param["num_epochs"]

    def get_batch(self, feed_data):
        X = feed_data["inputs"]
        Y = feed_data["labels"]
        totalbatch = int(len(X)/self.batch_size)+1
        if (totalbatch * self.batch_size == len(X)):
            totalbatch = totalbatch - 1

        for i in range(0,totalbatch):
            startindex = i*self.batch_size
            endindex = (i+1)*self.batch_size
            batch_xs = X[startindex:endindex]
            batch_ys = Y[startindex:endindex]
            yield { "batch_xs" : batch_xs, "batch_ys" : batch_ys }

    def train(self, feed_data):
        self.sess.run(tf.global_variables_initializer())
        trainstep = 0
        totallen = len(feed_data["inputs"])
        res = []

        for epoch in range(self.num_epochs):
            avg_cost = 0.0
            totalaccuracy = 0.0
            for batch in self.get_batch(feed_data):
                feed_dict = {
                    self.inputs : batch["batch_xs"],
                    self.labels : batch["batch_ys"],
                }
                _, loss, acc = self.sess.run([self.optimizer, self.loss,self.accuracy], feed_dict=feed_dict)
                totalaccuracy += acc*len(batch["batch_xs"])
                avg_cost +=  loss
                trainstep = trainstep + 1
            avg_cost /= totallen
            totalaccuracy /= len(feed_data['inputs'])
            dic = {"train_step":trainstep,"epoch":epoch+1,"accuracy":totalaccuracy,"loss":avg_cost}
            res.append(dic)
            print("train_step"+"\t"+str(trainstep)+"\t"+"epoch:"+"\t"+str(epoch+1)+"\t"+"accuracy:"+"\t"+str(totalaccuracy)+"\t"+"loss:"+"\t"+str(avg_cost))
        return res

    def evaluate(self, feed_data):
        avg_loss = 0.0
        totalaccuracy = 0.0
        totallen = len(feed_data["inputs"])

        for batch in self.get_batch(feed_data):
            feed_dict = {
                self.inputs: batch["batch_xs"],
                self.labels: batch["batch_ys"],
            }
            loss, acc = self.sess.run([self.loss, self.accuracy], feed_dict=feed_dict)
            totalaccuracy += acc * len(batch["batch_xs"])
            avg_loss += loss
        avg_loss /= totallen
        totalaccuracy /= len(feed_data['inputs'])
        res = {"accuracy":totalaccuracy,"loss":avg_loss}
        return res


    def model_load(self,path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        return

    def model_save(self,path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)
        return

    def predict(self, feed_data):
        res = []
        for batch in self.get_batch(feed_data):
            feed_dict = {
                self.inputs: batch["batch_xs"]
            }

            pred = self.sess.run(self.pred, feed_dict=feed_dict)
            res.extend(pred.tolist())

        return res
