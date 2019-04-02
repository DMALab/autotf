from __future__ import division, print_function, absolute_import

from base_model import BaseModel
import tensorflow as tf
import numpy as np
import random

class SVM(BaseModel):
    default_param = {
        "metrics" : [],
        "optimizer" : "sgd",
        "learning_rate" : 1e-2,
        "batch_size" : 256,
        "num_epochs" : 10,
        "alpha":0.01,
        "L2":True,
    }

    def __init__(self, feature_num, param=None):
        self.feature_num = feature_num
        self.sess = tf.Session()
        if param:
            self.set_parameter(param)

    def build_model(self):
        with tf.variable_scope("svm"):
            self.inputs = tf.placeholder(shape=[None,self.feature_num],dtype=tf.float32)
            self.labels = tf.placeholder(shape=[None],dtype=tf.float32)
            self.SvmWeight = tf.Variable(tf.random_normal(shape=[self.feature_num,1]),dtype=tf.float32)
            self.SvmBias = tf.Variable(tf.random_normal(shape=[1,1]),dtype=tf.float32)
            self.pred = tf.subtract(tf.matmul(self.inputs,self.SvmWeight),self.SvmBias)

            self.l2_norm = tf.reduce_sum(tf.square(self.SvmWeight))


    def set_parameter(self, param):
        for name  in self.default_param:
            if name not in param:
                param[name] = self.default_param[name]

        self.build_model()
        # loss = max(0,1-pred*actual) + alpha * L2_norm(weight)^2
        #metrics = [self.get_metric(metric) for metric in param["metrics"]]
        #self.metrics = [metric_fun(self.output, self.ground_truth) for metric_fun in metrics]


        self.actualloss =  tf.reduce_mean(tf.maximum(tf.subtract(1.,tf.multiply(self.pred,self.labels)),0.))
        self.alpha = param['alpha']

        self.loss = tf.add(self.actualloss,tf.multiply(self.alpha,self.l2_norm))

        self.optimizer = self.optimizer_dict[param["optimizer"]]
        learning_rate = param["learning_rate"]
        self.optimizer = self.optimizer(learning_rate).minimize(self.loss)
        self.batch_size = param["batch_size"]
        self.num_epochs = param["num_epochs"]
        self.prediction = tf.sign(self.pred)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.prediction,self.labels),tf.float32))

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
                print(acc)
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

    def predict(self, feed_data):
        res = []
        for batch in self.get_batch(feed_data):
            feed_dict = {
                self.inputs: batch["batch_xs"]
            }

            pred = self.sess.run(self.pred, feed_dict=feed_dict)
            res.extend(pred.tolist())

        return res


    def model_load(self,path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        return

    def model_save(self,path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)
        return
