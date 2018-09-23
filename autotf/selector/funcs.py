import numpy as np
import tensorflow as tf

from autotf.model.logistic_regression import LogisticRegression
from autotf.model.GoogleNet import GoogleNet
from autotf.model.AlexNet import AlexNet
from autotf.model.vgg16 import Vgg16
from autotf.model.RandomForest import RandomForest
from autotf.model.LinearSvm import LinearSVM
from autotf.model.KernelSvm import KernelSVM

from sklearn.datasets import load_svmlight_file
from sklearn.model_selection import train_test_split


def logistic_func(cfg, FOUT, x, y, feature_num, class_num):
    FOUT.write(str(cfg))

    cfg = {k: cfg[k] for k in cfg if cfg[k]}

    x_train, x_valid, x_test = x[0], x[1], x[2]
    y_train, y_valid, y_test = y[0], y[1], y[2]

    x_train = np.reshape(x_train, [len(x_train), -1])
    x_valid = np.reshape(x_valid, [len(x_valid), -1])
    x_test = np.reshape(x_test, [len(x_test), -1])

    params = {}
    params["loss"] = cfg["logistic_loss"]
    params["optimizer"] = cfg["logistic_optimizer"]
    params["learning_rate"] = cfg["logistic_learning_rate"]
    params["batch_size"] = cfg["logistic_batch_size"]
    params["num_epochs"] = cfg["logistic_num_epochs"]

    clf = LogisticRegression(feature_num=feature_num, classnum=class_num)
    clf.set_parameter(params)
    train_data = {"inputs": x_train, "labels": y_train}
    clf.train(train_data)

    valid_data = {"inputs": x_valid, "labels": y_valid}
    valid_performance = clf.evaluate(valid_data)
    FOUT.write("  valid performance = " + str(valid_performance) + "\n")

    if x_test is not None:
        test_data = {"inputs": x_test, "labels": y_test}
        test_performance = clf.evaluate(test_data)
        FOUT.write("  test performance = " + str(test_performance) + "\n")
        FOUT.write("---------------------------------\n")

    FOUT.flush()

    return 1 - valid_performance["accuracy"]


def alexnet_func(cfg, FOUT, x, y, feature_num, class_num):
    FOUT.write(str(cfg))

    x_train, x_valid, x_test = x[0], x[1], x[2]
    y_train, y_valid, y_test = y[0], y[1], y[2]

    print("alexnet func")
    with tf.device("/cpu:0"):
        x_train = tf.image.rgb_to_grayscale(x_train)
        x_valid = tf.image.rgb_to_grayscale(x_valid)
        if x_test is not None:
            x_test = tf.image.rgb_to_grayscale(x_test)

        x_train = tf.image.resize_images(x_train, [28, 28])
        x_valid = tf.image.resize_images(x_valid, [28, 28])
        x_test = tf.image.resize_images(x_test, [28, 28])

        x_train = tf.reshape(x_train, [len(x[0]), -1])
        x_valid = tf.reshape(x_valid, [len(x[1]), -1])
        x_test = tf.reshape(x_test, [len(x[2]), -1])

        with tf.Session() as sess:
            x_train, x_valid, x_test = sess.run([x_train, x_valid, x_test])

    print("after cpu session")

    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    params = {}
    params["loss"] = cfg["alexnet_loss"]
    params["optimizer"] = cfg["alexnet_optimizer"]
    params["learning_rate"] = cfg["alexnet_learning_rate"]
    params["batch_size"] = cfg["alexnet_batch_size"]
    params["num_epochs"] = cfg["alexnet_num_epochs"]
    params["keep_prob"] = cfg["alexnet_keep_prob"]

    print("alexnet well done!!!!")
    # start training
    clf = AlexNet(784, 10)

    train_data = {"inputs": x_train, "labels": y_train}
    clf.set_parameter(params)
    clf.train(train_data)

    valid_data = {"inputs": x_valid, "labels": y_valid}
    valid_performance = clf.evaluate(valid_data)
    FOUT.write("  valid performance = " + str(valid_performance) + "\n")

    if x_test is not None:
        test_data = {"inputs": x_test, "labels": y_test}
        test_performance = clf.evaluate(test_data)
        FOUT.write("  test performance = " + str(test_performance) + "\n")
        FOUT.write("---------------------------------\n")

    FOUT.flush()

    return 1 - valid_performance["accuracy"]


def vgg16_func(cfg, FOUT, x, y, feature_num, class_num):
    FOUT.write(str(cfg))

    cfg = {k: cfg[k] for k in cfg if cfg[k]}

    params = dict()
    params["loss"] = cfg["vgg16_loss"]
    params["optimizer"] = cfg["vgg16_optimizer"]
    params["learning_rate"] = cfg["vgg16_learning_rate"]
    params["batch_size"] = cfg["vgg16_batch_size"]
    params["num_epochs"] = cfg["vgg16_num_epochs"]
    params["keep_prob"] = cfg["vgg16_keep_prob"]

    x_train, x_valid, x_test = x[0], x[1], x[2]
    y_train, y_valid, y_test = y[0], y[1], y[2]

    with tf.device("/cpu:0"):
        if x_train.shape[3] == 1:
            x_train = tf.image.grayscale_to_rgb(x_train)
            x_valid = tf.image.grayscale_to_rgb(x_valid)
            if x_test is not None:
                x_test = tf.image.rgb_to_grayscale(x_test)

        x_train = tf.image.resize_images(x_train, [224, 224])
        x_valid = tf.image.resize_images(x_valid, [224, 224])
        x_test = tf.image.resize_images(x_test, [224, 224])

        with tf.Session() as sess:
            x_train, x_valid, x_test = sess.run([x_train, x_valid, x_test])

    # start training
    clf = Vgg16(class_num)

    train_data = {"inputs": x_train, "labels": y_train}
    clf.set_parameter(params)
    clf.train(train_data)

    valid_data = {"inputs": x_valid, "labels": y_valid}
    valid_performance = clf.evaluate(valid_data)
    FOUT.write("  valid performance = " + str(valid_performance) + "\n")

    if x_test is not None:
        test_data = {"inputs": x_test, "labels": y_test}
        test_performance = clf.evaluate(test_data)
        FOUT.write("  test performance = " + str(test_performance) + "\n")
        FOUT.write("---------------------------------\n")

    FOUT.flush()

    return 1 - valid_performance["accuracy"]


def googlenet_func(cfg, FOUT, x, y, feature_num, class_num):
    FOUT.write(str(cfg))

    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    params = dict()
    params["loss"] = cfg["googlenet_loss"]
    params["optimizer"] = cfg["googlenet_optimizer"]
    params["learning_rate"] = cfg["googlenet_learning_rate"]
    params["batch_size"] = cfg["googlenet_batch_size"]
    params["num_epochs"] = cfg["googlenet_num_epochs"]

    x_train, x_valid, x_test = x[0], x[1], x[2]
    y_train, y_valid, y_test = y[0], y[1], y[2]

    with tf.device("/cpu:0"):
        if x_train.shape[3] == 1:
            x_train = tf.image.grayscale_to_rgb(x_train)
            x_valid = tf.image.grayscale_to_rgb(x_valid)
            x_test = tf.image.grayscale_to_rgb(x_test)

        x_train = tf.image.resize_images(x_train, [227, 227])
        x_valid = tf.image.resize_images(x_valid, [227, 227])
        x_test = tf.image.resize_images(x_test, [227, 227])

        with tf.Session() as sess:
            x_train, x_valid, x_test = sess.run([x_train, x_valid, x_test])

    # start training
    clf = GoogleNet(class_num)

    train_data = {"inputs": x_train, "labels": y_train}
    clf.set_parameter(params)
    clf.train(train_data)

    valid_data = {"inputs": x_valid, "labels": y_valid}
    valid_performance = clf.evaluate(valid_data)
    FOUT.write("  valid performance = " + str(valid_performance) + "\n")

    if x_test is not None:
        test_data = {"inputs": x_test, "labels": y_test}
        test_performance = clf.evaluate(test_data)
        FOUT.write("  test performance = " + str(test_performance) + "\n")
        FOUT.write("---------------------------------\n")

    FOUT.flush()

    return 1 - valid_performance["accuracy"]


def randomforest_func(cfg, FOUT, x, y, feature_num, class_num):
    FOUT.write(str(cfg))

    x_train, x_valid, x_test = x[0], x[1], x[2]
    y_train, y_valid, y_test = y[0], y[1], y[2]

    # x_train = np.reshape(x_train, [-1, feature_num])
    # x_valid = np.reshape(x_valid, [-1, feature_num])
    # x_test = np.reshape(x_test, [-1, feature_num])
    #
    # y_train = np.argmax(y_train, axis=1)
    # y_valid = np.argmax(y_valid, axis=1)
    # y_test = np.argmax(y_test, axis=1)

    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    params = dict({"loss": "square_loss", "metrics": ["loss"], "max_nodes": 1000})
    params["batch_size"] = cfg["randomforest_batch_size"]
    params["num_epochs"] = cfg["randomforest_num_epochs"]
    params["num_trees"] = cfg["randomforest_num_trees"]

    model = RandomForest(feature_num, class_num)

    train_data = {"inputs": x_train, "labels": y_train}
    model.set_parameter(params)
    model.train(train_data)

    valid_data = {"inputs": x_valid, "labels": y_valid}
    valid_performance = model.evaluate(valid_data)
    FOUT.write("  valid performance = " + str(valid_performance) + "\n")

    if x_test is not None:
        test_data = {"inputs": x_test, "labels": y_test}
        test_performance = model.evaluate(test_data)
        FOUT.write("  test performance = " + str(test_performance) + "\n")
        FOUT.write("---------------------------------\n")
    FOUT.flush()

    return 1 - valid_performance["accuracy"]


def linearsvm_func(cfg, FOUT, x, y, feature_num, class_num):
    FOUT.write(str(cfg))

    x_train, x_valid, x_test = x[0], x[1], x[2]
    y_train, y_valid, y_test = y[0], y[1], y[2]

    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    params = dict({"metrics": [], "class_num": class_num})
    params["batch_size"] = cfg["linearsvm_batch_size"]
    params["num_epochs"] = cfg["linearsvm_num_epochs"]

    model = LinearSVM(feature_num)

    train_data = {"inputs": x_train, "labels": y_train}
    model.set_parameter(params)
    model.train(train_data)

    valid_data = {"inputs": x_valid, "labels": y_valid}
    valid_performance = model.evaluate(valid_data)
    FOUT.write("  valid performance = " + str(valid_performance) + "\n")

    if x_test is not None:
        test_data = {"inputs": x_test, "labels": y_test}
        test_performance = model.evaluate(test_data)
        FOUT.write("  test performance = " + str(test_performance) + "\n")
        FOUT.write("---------------------------------\n")
    FOUT.flush()

    return 1 - valid_performance["accuracy"]


def kernelsvm_func(cfg, FOUT, x, y, feature_num, class_num):
    FOUT.write(str(cfg))

    x_train, x_valid, x_test = x[0], x[1], x[2]
    y_train, y_valid, y_test = y[0], y[1], y[2]

    cfg = {k: cfg[k] for k in cfg if cfg[k]}
    params = dict({
        "metrics": [],
        "class_num": class_num,
        "l2": 0.001,
        "stddev": 5.0,
        "out_dim": 2000,
    })
    params["batch_size"] = cfg["kernelsvm_batch_size"]
    params["num_epochs"] = cfg["kernelsvm_num_epochs"]
    params["learning_rate"] = cfg["kernelsvm_learning_rate"]

    model = KernelSVM(feature_num)

    train_data = {"inputs": x_train, "labels": y_train}
    model.set_parameter(params)
    model.train(train_data)

    valid_data = {"inputs": x_valid, "labels": y_valid}
    valid_performance = model.evaluate(valid_data)
    FOUT.write("  valid performance = " + str(valid_performance) + "\n")

    if x_test is not None:
        test_data = {"inputs": x_test, "labels": y_test}
        test_performance = model.evaluate(test_data)
        FOUT.write("  test performance = " + str(test_performance) + "\n")
        FOUT.write("---------------------------------\n")
    FOUT.flush()

    return 1 - valid_performance["accuracy"]
