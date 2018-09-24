from __future__ import division, print_function, absolute_import

from base_model import BaseModel
from keras.preprocessing import sequence
import tensorflow as tf
import numpy as np
import random
import pandas as pd
import re
from tensorflow.examples.tutorials.mnist import input_data
from tflearn.data_utils import to_categorical, pad_sequences
from Rnn import *
from keras.datasets import imdb
import time
from keras.utils.np_utils import to_categorical

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
def GetData():
    max_features = 20000
    maxlen = 80  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32

    print('Loading data...\n')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences\n')
    print(len(x_test), 'test sequences\n')

    print('Pad sequences (samples x time)\n')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    y_train = to_categorical(y_train,2)
    y_test = to_categorical(y_test, 2)
    print('x_train shape:\t', x_train.shape)
    print('x_test shape:\t', x_test.shape)
    return max_features,x_train,x_test,y_train,y_test,batch_size,maxlen

def GetData2():
    maxlen = 100
    df = pd.read_csv("/home/share/rnndata/Toxic/train.csv")
    dftest = pd.read_csv("/home/share/rnndata/Toxic/test.csv")
    sentences = df['comment_text'].tolist()
    testsentence = dftest['comment_text'].tolist()
    testlabels = pd.read_csv("/home/share/rnndata/Toxic/test_labels.csv")
    dftest = dftest.merge(testlabels, on="id", how="left")

    def GetLabel(idx, df):
        array = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        labels = df[array[idx]].tolist()
        return labels

    def GetTestIndex(testsentence, idx, df):
        array = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
        labels = df[array[idx]].tolist()
        reslabels = []
        ressentence = []
        for i in range(0, len(labels)):
            if (labels[i] == -1):
                continue
            else:
                reslabels.append(labels[i])
                ressentence.append(testsentence[i])
        return ressentence, reslabels

    def ProcessSentence(sentences):
        rule = re.compile("[^a-zA-Z0-9]")
        for i in range(0, len(sentences)):
            sentences[i] = rule.sub(" ", str(sentences[i]))
        return sentences

    def GetDictionary(sentences, threshold=5):
        dictionary = {}
        for line in sentences:
            tokens = line.split(" ")
            for token in tokens:
                if token == "":
                    continue
                if token not in dictionary:
                    dictionary[token] = 1
                else:
                    dictionary[token] = dictionary[token] + 1
        newdictionary = {}
        idx = 2
        for token in dictionary:
            if dictionary[token] > threshold:
                newdictionary[token] = idx
                idx = idx + 1

        MaxTokens = len(newdictionary)
        return newdictionary, MaxTokens

    def BuildNumber(sentences, dictionary):
        result = []
        for line in sentences:
            curline = []
            tokens = line.split(" ")
            for token in tokens:
                if token == "":
                    continue
                if token in dictionary:
                    curline.append(dictionary[token])
                else:
                    curline.append(1)
            result.append(curline)
        return result

    sentences = ProcessSentence(sentences)
    testsentence = ProcessSentence(testsentence)
    testsentence, TestY = GetTestIndex(testsentence, 0, dftest)
    dictionary, MaxTokens = GetDictionary(sentences)
    TrainX = BuildNumber(sentences, dictionary)
    TestX = BuildNumber(testsentence, dictionary)
    TrainY = GetLabel(0, df)
    TrainX = pad_sequences(TrainX, maxlen=maxlen, value=0.)
    TestX = pad_sequences(TestX, maxlen=maxlen, value=0.)
    TrainX = np.array(TrainX)
    TrainY = np.array(TrainY)
    TestX = np.array(TestX)
    TestY = np.array(TestY)
    return maxlen,TrainX,TestX,TrainY,TestY

def TestRnn(epochnumber):
    tf.reset_default_graph()
    max_features, x_train, x_test, y_train, y_test, batch_size, maxlen = GetData()
    print('Build model Rnn...')
    params = {
        "metrics": [],
        "batch_size": batch_size,
        "class_num": 2,
        "num_epochs": epochnumber,
        "learning_rate": 1e-3,
        "hidden_dimension": 128,
        "sentence_len":maxlen,
        "embdding_dimension":128,
        "vocab_size":max_features,
        "layer_num":1,
        "CellName":"RNNCell",
        "IsBidirection": False
    }

    m = RnnModel()
    m.set_parameter(params)
    feed_data = {"inputs": x_train, "labels": y_train}
    test_feed_data = {"inputs": x_test, "labels": y_test}
    ts = time.time()
    m.train(feed_data, test_feed_data)
    te = time.time()
    costtime  = te-ts
    print("time cost\t" + str(costtime))
    dic = m.evaluate(test_feed_data)
    print(dic)
    m.model_save("/home/share/model/RnnModel.ckpt")

def TestLstm(epochnumber):
    tf.reset_default_graph()
    max_features, x_train, x_test, y_train, y_test, batch_size, maxlen = GetData()
    print('Build model LSTM...\n')
    params = {
        "metrics": [],
        "batch_size": batch_size,
        "class_num": 2,
        "num_epochs": epochnumber,
        "learning_rate": 1e-3,
        "hidden_dimension": 128,
        "sentence_len":maxlen,
        "embdding_dimension":128,
        "vocab_size":max_features,
        "layer_num":1,
        "CellName":"LSTMCell",
        "IsBidirection": False
    }

    m = RnnModel()
    m.set_parameter(params)
    feed_data = {"inputs": x_train, "labels": y_train}
    test_feed_data = {"inputs": x_test, "labels": y_test}
    ts = time.time()
    m.train(feed_data, test_feed_data)
    te = time.time()
    costtime  = te-ts
    print("time cost\t" + str(costtime))
    dic =  m.evaluate(test_feed_data)
    print(dic)
    m.model_save("/home/share/model/LstmModel.ckpt")
    return dic


def TestGRU(epochnumber):
    tf.reset_default_graph()
    max_features, x_train, x_test, y_train, y_test, batch_size, maxlen = GetData()
    print('Build model GRU...')
    params = {
        "metrics": [],
        "batch_size": batch_size,
        "class_num": 2,
        "num_epochs": epochnumber,
        "learning_rate": 1e-3,
        "hidden_dimension": 128,
        "sentence_len":maxlen,
        "embdding_dimension":128,
        "vocab_size":max_features,
        "layer_num":1,
        "CellName":"GRUCell",
        "IsBidirection":False
    }

    m = RnnModel()
    m.set_parameter(params)
    feed_data = {"inputs": x_train, "labels": y_train}
    test_feed_data = {"inputs": x_test, "labels": y_test}
    ts = time.time()
    m.train(feed_data, test_feed_data)
    te = time.time()
    costtime  = te-ts
    print("time cost\t" + str(costtime))
    dic =  m.evaluate(test_feed_data)
    print(dic)
    m.model_save("/home/share/model/GRUModel.ckpt")


def TestBiRnn(epochnumber):
    tf.reset_default_graph()
    max_features, x_train, x_test, y_train, y_test, batch_size, maxlen = GetData()

    params = {
        "metrics": [],
        "batch_size": batch_size,
        "class_num": 2,
        "num_epochs": epochnumber,
        "learning_rate": 1e-2,
        "hidden_dimension": 128,
        "sentence_len":maxlen,
        "embdding_dimension":128,
        "vocab_size":max_features,
        "layer_num":1,
        "CellName":"RNNCell",
        "IsBidirection": True
    }

    m = RnnModel()
    m.set_parameter(params)
    feed_data = {"inputs": x_train, "labels": y_train}
    test_feed_data = {"inputs": x_test, "labels": y_test}
    ts = time.time()
    m.train(feed_data, test_feed_data)
    te = time.time()
    costtime  = te-ts
    print("time cost\t" + str(costtime))
    dic = m.evaluate(test_feed_data)
    print(dic)
    m.model_save("/home/share/model/RnnModel.ckpt")


def TestBiLstm(epochnumber):
    tf.reset_default_graph()
    max_features, x_train, x_test, y_train, y_test, batch_size, maxlen = GetData()
    params = {
        "metrics": [],
        "batch_size": batch_size,
        "class_num": 2,
        "num_epochs": epochnumber,
        "learning_rate": 1e-3,
        "hidden_dimension": 128,
        "sentence_len":maxlen,
        "embdding_dimension":128,
        "vocab_size":max_features,
        "layer_num":1,
        "CellName":"LSTMCell",
        "IsBidirection": True
    }

    m = RnnModel()
    m.set_parameter(params)
    feed_data = {"inputs": x_train, "labels": y_train}
    test_feed_data = {"inputs": x_test, "labels": y_test}
    ts = time.time()
    m.train(feed_data, test_feed_data)
    te = time.time()
    costtime  = te-ts
    print("time cost\t" + str(costtime))
    dic =  m.evaluate(test_feed_data)
    print(dic)
    m.model_save("/home/share/model/LstmModel.ckpt")
    m.model_load("/home/share/model/LstmModel.ckpt")


def TestBiGRU(epochnumber):
    tf.reset_default_graph()
    max_features, x_train, x_test, y_train, y_test, batch_size, maxlen = GetData()
    params = {
        "metrics": [],
        "batch_size": batch_size,
        "class_num": 2,
        "num_epochs": epochnumber,
        "learning_rate": 1e-3,
        "hidden_dimension": 128,
        "sentence_len":maxlen,
        "embdding_dimension":128,
        "vocab_size":max_features,
        "layer_num":1,
        "CellName":"GRUCell",
        "IsBidirection":True
    }

    m = RnnModel()
    m.set_parameter(params)
    feed_data = {"inputs": x_train, "labels": y_train}
    test_feed_data = {"inputs": x_test, "labels": y_test}
    ts = time.time()
    m.train(feed_data, test_feed_data)
    te = time.time()
    costtime  = te-ts
    print("time cost\t" + str(costtime))
    dic =  m.evaluate(test_feed_data)
    print(dic)
    m.model_save("/home/share/model/GRUModel.ckpt")


epochnumber = 15
TestRnn(epochnumber)
TestGRU(epochnumber)
TestLstm(epochnumber)
TestBiRnn(epochnumber)
TestBiLstm(epochnumber)
TestBiGRU(epochnumber)