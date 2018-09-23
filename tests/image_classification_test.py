import pickle
import os
import numpy as np
import tensorflow as tf

from autotf.selector.accurate_selector import AccurateSelector
# from autotf.model.AlexNet import AlexNet
# from autotf.model.GoogleNet import GoogleNet
# from autotf.model.logistic_regression import LogisticRegression
from sklearn.model_selection import train_test_split
from autotf.model.RandomForest import RandomForest


def unpickle(file):
    with open(file, "rb") as fo:
        d = pickle.load(fo, encoding="bytes")
    return d


def alexnet_test(x, y):
    x_train = x[0]
    y_train = y[0]

    x_valid = x[1]
    y_valid = y[1]

    with tf.device("/cpu:0"):
        x_train = tf.image.rgb_to_grayscale(x_train)
        x_valid = tf.image.rgb_to_grayscale(x_valid)

        x_train = tf.image.resize_images(x_train, [28, 28])
        x_valid = tf.image.resize_images(x_valid, [28, 28])

        x_train = tf.reshape(x_train, [len(x[0]), -1])
        x_valid = tf.reshape(x_valid, [len(x[1]), -1])

        with tf.Session() as sess:
            x_train, x_valid = sess.run([x_train, x_valid])


    params = {
        "loss": "square_loss",
        "metrics": ["loss"],
        "optimizer": "sgd",
        "learning_rate": 5e-4,
        "batch_size": 20,
        "num_epochs": 10,
        "keep_prob": 0.75
    }
    m = AlexNet(784, 10)
    m.set_parameter(params)

    train_data = {"inputs": x_train, "labels": y_train}
    m.train(train_data)

    valid_data = {"inputs": x_valid, "labels": y_valid}
    dic = m.evaluate(valid_data)

    print("Evaluate:" + str(dic))


def googlenet_test(x, y):

    x_train, x_valid = x[0], x[1]
    y_train, y_valid = y[0], y[1]

    with tf.device("/cpu:0"):
        # x_train = tf.image.grayscale_to_rgb(x_train)
        # x_valid = tf.image.grayscale_to_rgb(x_valid)
        # x_test = tf.image.grayscale_to_rgb(x_test)

        x_train = tf.image.resize_images(x_train, [227, 227])
        x_valid = tf.image.resize_images(x_valid, [227, 227])

        with tf.Session() as sess:
            x_train, x_valid = sess.run([x_train, x_valid])

    m = GoogleNet(10)

    train_data = {"inputs": x_train, "labels": y_train}
    valid_data = {"inputs": x_valid, "labels": y_valid}

    default_param = {
                        "loss" : "square_loss",
                        "metrics" : ["loss"],
                        "optimizer" : "sgd",
                        "learning_rate" : 5e-4,
                        "batch_size" : 50,
                        "num_epochs" : 20,
                    }

    m.set_parameter(default_param)

    m.train(train_data)

    dic = m.evaluate(valid_data)
    print("Evaluate:"+str(dic))


def logistic_test(x, y):
    x_train, x_valid = x[0], x[1]
    y_train, y_valid = y[0], y[1]

    x_train = np.reshape(x_train, [len(x_train), -1])
    x_valid = np.reshape(x_valid, [len(x_valid), -1])

    m = LogisticRegression(32 * 32 * 3, 10)
    params = {
        "loss": "square_loss",
        "metrics": ["loss"],
        "optimizer": "sgd",
        "learning_rate": 3e-4,
        "batch_size": 20,
        "num_epochs": 10
    }
    m.set_parameter(params)
    train_data = {"inputs": x_train, "labels": y_train}
    m.train(train_data)

    valid_data = {"inputs": x_valid, "labels": y_valid}
    dic = m.evaluate(valid_data)
    print("Evaluate:" + str(dic))


def vgg16_test(x, y):
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


def randomforest_test(x, y):
    x_train, x_test = x
    y_train, y_test = y

    train_num = len(x_train)
    test_num = len(x_test)
    x_train = np.reshape(x_train, [train_num, -1])
    x_test = np.reshape(x_test, [test_num, -1])

    m = RandomForest(3072, 10)

    params = {
        "loss": "square_loss",
        "metrics": ["loss"],
        "batch_size": 20,
        "num_epochs": 10,
        "num_trees": 10,
        "max_nodes": 1000
    }

    m.set_parameter(params)

    train_data = {"inputs": x_train, "labels": y_train}
    m.train(feed_data=train_data)

    test_data = {"inputs": x_test, "labels": y_test}
    dic = m.evaluate(feed_data=test_data)
    print("RandomForest Evaluate:" + str(dic))


# data_path = "/Users/xuehuanran/datasets/cifar10/cifar10.npz"
# data_path = "/home/daim_gpu/xuehuanran/datasets/cifar10/cifar10.npz"
# data_path = "/home/daim_gpu/xuehuanran/datasets/cifar10/cifar-10-batches-py/"
data_path = "/Users/xuehuanran/datasets/cifar10/cifar-10-batches-py/"


x_train = np.load(data_path + "x_train.npy")
y_train = np.load(data_path + "y_train.npy")
x_valid = np.load(data_path + "x_valid.npy")
y_valid = np.load(data_path + "y_valid.npy")
x_test = np.load(data_path + "x_test.npy")
y_test = np.load(data_path + "y_test.npy")


n_samples = 1000
samples = np.random.choice(len(x_train), n_samples, replace=False)
x_train, y_train = x_train[samples], y_train[samples]

print(x_train.shape)
print(y_train.shape)
print(x_valid.shape)
print(y_valid.shape)
print(x_test.shape)
print(y_test.shape)

selector = AccurateSelector(task_type="image_classification")
selector.select_model([x_train, x_valid, x_test], [y_train, y_valid, y_test], feature_num=32 * 32 * 3, class_num=10)
