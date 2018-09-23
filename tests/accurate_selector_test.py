import os
import tensorflow as tf
import numpy as np

from autotf.model.AlexNet import AlexNet
from autotf.model.GoogleNet import GoogleNet
from autotf.model.logistic_regression import LogisticRegression
from autotf.model.vgg16 import Vgg16
from tensorflow.examples.tutorials.mnist import input_data

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

mnist = input_data.read_data_sets("/home/daim_gpu/xuehuanran/datasets/mnist", one_hot=True)
# mnist = input_data.read_data_sets("/Users/xuehuanran/datasets/mnist", one_hot=True)


def alexnet_test():
    x_train = mnist.train.images
    y_train = mnist.train.labels

    x_valid = mnist.validation.images
    y_valid = mnist.validation.labels

    x_test = mnist.test.images
    y_test = mnist.test.labels

    params = {
        "loss": "square_loss",
        "metrics": ["loss"],
        "optimizer": "sgd",
        "learning_rate": 5e-4,
        "batch_size": 100,
        "num_epochs": 2,
        "keep_prob": 0.75
    }
    m = AlexNet(784, 10)
    m.set_parameter(params)

    train_data = {"inputs": x_train, "labels": y_train}
    m.train(train_data)

    valid_data = {"inputs": x_valid, "labels": y_valid}
    dic = m.evaluate(valid_data)
    print("Evaluate:"+str(dic))

    test_data = {"inputs": x_test, "labels": y_test}
    dic = m.evaluate(test_data)
    print("Evaluate:"+str(dic))


def googlenet_test():
    x_train = mnist.train.images
    y_train = mnist.train.labels

    x_valid = mnist.validation.images
    y_valid = mnist.validation.labels

    x_test = mnist.test.images
    y_test = mnist.test.labels

    m = GoogleNet(10)

    x_train = np.reshape(x_train, [-1, 28, 28, 1])
    x_valid = np.reshape(x_valid, [-1, 28, 28, 1])
    x_test = np.reshape(x_test, [-1, 28, 28, 1])

    with tf.device("/cpu:0"):
        x_train = tf.image.grayscale_to_rgb(x_train)
        x_valid = tf.image.grayscale_to_rgb(x_valid)
        x_test = tf.image.grayscale_to_rgb(x_test)

        x_train = tf.image.resize_images(x_train, [227, 227])
        x_valid = tf.image.resize_images(x_valid, [227, 227])
        x_test = tf.image.resize_images(x_test, [227, 227])

        with tf.Session() as sess:
            x_train, x_valid, x_test = sess.run([x_train, x_valid, x_test])

    train_data = {"inputs": x_train, "labels": y_train}
    valid_data = {"inputs": x_valid, "labels": y_valid}
    test_data = {"inputs": x_test, "labels": y_test}

    default_param = {
                        "loss" : "square_loss",
                        "metrics" : ["loss"],
                        "optimizer" : "sgd",
                        "learning_rate" : 1e-3,
                        "batch_size" : 50,
                        "num_epochs" : 2,
                    }

    m.set_parameter(default_param)

    m.train(train_data)

    dic = m.evaluate(valid_data)
    print("Evaluate:"+str(dic))

    dic = m.evaluate(test_data)
    print("Evaluate:"+str(dic))


def logistic_test():
    x_train = mnist.train.images
    y_train = mnist.train.labels

    x_valid = mnist.validation.images
    y_valid = mnist.validation.labels

    x_test = mnist.test.images
    y_test = mnist.test.labels

    m = LogisticRegression(784, 10)
    params = {
        "loss": "square_loss",
        "metrics": ["loss"],
        "optimizer": "sgd",
        "learning_rate": 3e-4,
        "batch_size": 20,
        "num_epochs": 2
    }
    m.set_parameter(params)
    train_data = {"inputs": x_train, "labels": y_train}
    m.train(train_data)

    valid_data = {"inputs": x_valid, "labels": y_valid}
    test_data = {"inputs": x_test, "labels": y_test}

    dic = m.evaluate(valid_data)
    print("Evaluate:" + str(dic))

    dic = m.evaluate(test_data)
    print("Evaluate:" + str(dic))


def vgg16_test():
    x_train = mnist.train.images
    y_train = mnist.train.labels

    x_valid = mnist.validation.images
    y_valid = mnist.validation.labels

    x_test = mnist.test.images
    y_test = mnist.test.labels

    x_train = np.reshape(x_train, [-1, 28, 28, 1])
    x_valid = np.reshape(x_valid, [-1, 28, 28, 1])
    x_test = np.reshape(x_test, [-1, 28, 28, 1])

    with tf.device("/cpu:0"):
        x_train = tf.image.grayscale_to_rgb(x_train)
        x_valid = tf.image.grayscale_to_rgb(x_valid)
        x_test = tf.image.grayscale_to_rgb(x_test)

        x_train = tf.image.resize_images(x_train, [224, 224])
        x_valid = tf.image.resize_images(x_valid, [224, 224])
        x_test = tf.image.resize_images(x_test, [224, 224])

        with tf.Session() as sess:
            x_train, x_valid, x_test = sess.run([x_train, x_valid, x_test])

    m = Vgg16(10)
    params = {
        "loss": "square_loss",
        "metrics": ["loss"],
        "optimizer": "sgd",
        "learning_rate": 1e-4,
        "batch_size": 32,
        "num_epochs": 2,
        "keep_prob": 0.75
    }
    m.set_parameter(params)
    train_data = {"inputs": x_train, "labels": y_train}
    m.train(train_data)

    valid_data = {"inputs": x_valid, "labels": y_valid}
    test_data = {"inputs": x_test, "labels": y_test}

    dic = m.evaluate(valid_data)
    print("Evaluate:" + str(dic))

    dic = m.evaluate(test_data)
    print("Evaluate:" + str(dic))


if __name__ == '__main__':
    alexnet_test()
    # googlenet_test()
    logistic_test()
    # vgg16_test()
