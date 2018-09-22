from __future__ import print_function
import os
from keras.preprocessing import sequence
from keras.models import Sequential
import keras.backend as K
from keras.optimizers import Adam
from keras.layers import Dropout
from keras.layers import Dense, Embedding
from keras.layers import LSTM
from keras.layers import SimpleRNN
from keras.layers import GRU
from keras.layers import  Bidirectional
from keras.datasets import imdb
from keras.utils.np_utils import to_categorical

import time

os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def GetData():
    max_features = 20000
    maxlen = 80  # cut texts after this number of words (among top max_features most common words)
    batch_size = 32
    droupoutrate = 0.20

    print('Loading data...\n')
    (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_features)
    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')

    print('Pad sequences (samples x time)\n')
    x_train = sequence.pad_sequences(x_train, maxlen=maxlen)
    x_test = sequence.pad_sequences(x_test, maxlen=maxlen)
    y_train = to_categorical(y_train,2)
    y_test = to_categorical(y_test, 2)
    print('x_train shape:\t', x_train.shape)
    print('x_test shape:\t', x_test.shape)

    return max_features,x_train,x_test,y_train,y_test,batch_size,maxlen,droupoutrate


def preprocess(name,max_features,maxlen,em_dimension=128):
    print('Build model ' + name + '...\n')
    model = Sequential()
    model.add(Embedding(max_features, em_dimension,input_length=maxlen))
    return model

def postpreprocess(model, x_train,x_test, y_train,y_test, batch_size, epoch,droupoutrate, loss_name='categorical_crossentropy'):
    model.add(Dropout(droupoutrate))
    model.add(Dense(2,activation='softmax'))

    model.compile(loss=loss_name,
              optimizer="Adam",
              metrics=['accuracy'])

    print('Start Train...\n')
    ts = time.time()
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epoch,verbose=0)
    te = time.time()
    cost_time  = te-ts
    print("time cost\t" + str(cost_time))
    score, acc = model.evaluate(x_test, y_test,
                                batch_size=batch_size,verbose=0)
    trainscore, trainacc = model.evaluate(x_train, y_train,
                                batch_size=batch_size,verbose=0)

    print("Train score:\t",trainscore)
    print("Train accuracy:\t",trainacc)
    print('Test score:\t', score)
    print('Test accuracy:\t', acc)
    print("\n")
    return


def TestRNN(name,epoch):
    max_features, x_train, x_test, y_train, y_test, batch_size, maxlen, droupoutrate = GetData()
    model = preprocess(name,max_features,maxlen,128)
    model.add(SimpleRNN(units=128))
    postpreprocess(model, x_train,x_test, y_train,y_test, batch_size, epoch,droupoutrate, loss_name='categorical_crossentropy')
    return


def TestLSTM(name,epoch):
    max_features, x_train, x_test, y_train, y_test, batch_size, maxlen, droupoutrate = GetData()
    model = preprocess(name,max_features,maxlen,128)
    model.add(LSTM(units=128))
    postpreprocess(model,x_train,x_test,y_train,y_test,batch_size,epoch,droupoutrate,'categorical_crossentropy')
    return


def TestGRU(name,epoch):
    max_features, x_train, x_test, y_train, y_test, batch_size, maxlen, droupoutrate = GetData()
    model = preprocess(name,max_features,maxlen,128)
    model.add(GRU(units=128))
    postpreprocess(model, x_train, x_test, y_train, y_test, batch_size, epoch,droupoutrate, 'categorical_crossentropy')
    return


def TestBiRnn(name,epoch):
    max_features, x_train, x_test, y_train, y_test, batch_size, maxlen, droupoutrate = GetData()
    model = preprocess(name,max_features,maxlen,128)
    model.add(Bidirectional(SimpleRNN(units = 128),merge_mode="concat"))
    postpreprocess(model, x_train, x_test, y_train, y_test, batch_size, epoch,droupoutrate, 'categorical_crossentropy')
    return


def TestBiLSTM(name,epoch):
    max_features, x_train, x_test, y_train, y_test, batch_size, maxlen, droupoutrate = GetData()
    model = preprocess(name,max_features,maxlen,128)
    model.add(Bidirectional(LSTM(units = 128),merge_mode="concat"))
    postpreprocess(model, x_train, x_test, y_train, y_test, batch_size, epoch,droupoutrate, 'categorical_crossentropy')
    return


def TestBiGru(name,epoch):
    max_features, x_train, x_test, y_train, y_test, batch_size, maxlen, droupoutrate = GetData()
    model = preprocess(name,max_features,maxlen,128)
    model.add(Bidirectional(LSTM(units = 128),merge_mode="concat"))
    postpreprocess(model, x_train, x_test, y_train, y_test, batch_size, epoch,droupoutrate, 'categorical_crossentropy')
    return


epoch = 1


TestRNN("RNN",epoch)
TestLSTM("LSTM",epoch)
TestGRU("GRU",epoch)
TestBiRnn("Bidirectional Rnn",epoch)
TestBiLSTM("Bidirectional LSTM",epoch)
TestBiGru("Bidirectional Gru",epoch)
