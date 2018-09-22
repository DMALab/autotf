import tensorflow as tf
import numpy as np

class RnnModel():

    def __init__(self):
        self.sess = tf.Session()

    def GetCell(self,name):
        if name is "RNNCell":
            return tf.nn.rnn_cell.BasicRNNCell(self.hidden_dimension)
        if name is "LSTMCell":
            return tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dimension)
        if name is "GRUCell":
            return tf.nn.rnn_cell.GRUCell(self.hidden_dimension)
        return None
    def GetBiCell(self,name,inputs):
        #inputs = tf.transpose(inputs,[1,0,2])
        # now the inputs is [times,batchsize,embeedingsize]

        #inputs = tf.unstack(inputs)
        # inputs is [[batchsize,embeedingsize],[batchsize,embeedingsize],...[batchsize,embeedingsize]]

        self.fwcell = self.GetCell(name)
        self.bgcell = self.GetCell(name)

        #the inputs is [batch,timestep,dimension]
        outputs,_ = tf.nn.bidirectional_dynamic_rnn(self.fwcell,self.bgcell,inputs,sequence_length=self.source_sentence_length,dtype=tf.float32)

        # outputs is the [2,batch_size,sentence_length,hidden_dimension]
        outputs = tf.concat(outputs,2)

        print(outputs.get_shape())
        return outputs
    def set_parameter(self, param):
        self.layer_num = param["layer_num"]
        self.embdding_dimension = param["embdding_dimension"]
        self.vocab_size = param["vocab_size"]
        self.sentence_len = param["sentence_len"]
        self.hidden_dimension = param["hidden_dimension"]
        self.learning_rate = param["learning_rate"]
        self.num_epochs = param["num_epochs"]
        self.batch_size = param["batch_size"]
        self.class_num = param["class_num"]
        self.cellname = param["CellName"]

        self.inputs = tf.placeholder(tf.int32, shape=[None, self.sentence_len])
        self.labels = tf.placeholder(tf.int32, shape=[None,self.class_num])
        self.keep_prob = tf.placeholder(tf.float32)
        self.source_sentence_length = tf.placeholder(tf.int32,shape=[None])

        self.embedding_weight = tf.Variable(tf.truncated_normal((self.vocab_size, self.embdding_dimension)))

        self.embedding = tf.nn.embedding_lookup(self.embedding_weight, self.inputs)
        #self.embedding shape is [batch,timestep,embedding_length]

        if param["IsBidirection"]:
            self.outputs = self.GetBiCell(self.cellname,self.embedding)
        else:
            self.cell = self.GetCell(self.cellname)
            self.rnn = tf.nn.rnn_cell.MultiRNNCell([ self.cell  for _ in range(self.layer_num)])

            # the outputs means [BatchSize,timestate,hidden_number]
            self.outputs, _ = tf.nn.dynamic_rnn(self.rnn, self.embedding, dtype=tf.float32)

        #self.cell = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_dimension)

        # the pooling means that get the sum of the  dimension 1=> add all timestate
        self.pooling = tf.reduce_sum(self.outputs, 1)

        self.dropout = tf.nn.dropout(self.pooling, keep_prob=self.keep_prob)
        if param["IsBidirection"]:
            self.fcweight = tf.get_variable("fcweight",shape=[self.hidden_dimension*2,self.class_num],initializer=tf.contrib.layers.xavier_initializer())
            self.fcbias = tf.get_variable("fcbias",shape=[self.class_num],initializer=tf.constant_initializer(0.0))
        else:
            self.fcweight = tf.get_variable("fcweight",shape=[self.hidden_dimension,self.class_num],initializer=tf.contrib.layers.xavier_initializer())
            self.fcbias = tf.get_variable("fcbias",shape=[self.class_num],initializer=tf.constant_initializer(0.0))

        #self.prediction = tf.nn.softmax(tf.matmul(self.dropout, self.fcweight) + self.fcbias)
        self.logits = tf.nn.bias_add(tf.matmul(self.dropout, self.fcweight),self.fcbias)
        self.prediction = tf.nn.softmax(self.logits)

        self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.labels))

        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss)

        self.correct_prediction = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.labels, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))

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

    def train(self, feed_data,test_data=None,UseEvaluate=False):
        self.sess.run(tf.global_variables_initializer())
        trainstep = 0
        trainauc = []
        trainloss = []
        test = []
        for epoch in range(0,self.num_epochs):
            avg_cost = 0.0
            totalaccuracy = 0.0

            for batch in self.get_batch(feed_data):
                feed_dict = {
                    self.inputs : batch["batch_xs"],
                    self.labels : batch["batch_ys"],
                    self.keep_prob: 0.8,
                    self.source_sentence_length: [self.sentence_len]*len(batch["batch_xs"]),
                }
                #self.sess.run(self.optimizer,feed_dict=feed_dict)
                _, loss, train_accuracy = self.sess.run([self.optimizer, self.loss,self.accuracy], feed_dict=feed_dict)
                totalaccuracy +=  train_accuracy*len(batch["batch_xs"])
                avg_cost +=  loss
                trainstep = trainstep + 1
            totalaccuracy /= len(feed_data["inputs"])
            avg_cost /=  len(feed_data["inputs"])

            if UseEvaluate:
                testdic = self.evaluate(test_data)
                test.append(testdic)
                print(testdic)
            trainauc.append(totalaccuracy)
            trainloss.append(avg_cost)
            print("train_step"+"\t"+str(trainstep)+"\t"+"epoch:"+"\t"+str(epoch+1)+"\t"+"accuracy:"+"\t"+str(totalaccuracy)+"\t"+"loss:"+"\t"+str(avg_cost))

        return trainauc,trainloss,test

    def evaluate(self, feed_data):
        if feed_data is None:
            return {}
        avg_loss = 0.0
        totalaccuracy = 0.0
        totallen = len(feed_data["inputs"])

        for batch in self.get_batch(feed_data):
            feed_dict = {
                self.inputs: batch["batch_xs"],
                self.labels: batch["batch_ys"],
                self.keep_prob: 1.0,
                self.source_sentence_length: [self.sentence_len] * len(batch["batch_xs"]),
            }
            pred, loss, acc = self.sess.run([self.prediction, self.loss, self.accuracy], feed_dict=feed_dict)
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
                self.inputs: batch["batch_xs"],
                self.keep_prob: 1.0,
                self.source_sentence_length: [self.sentence_len] * len(batch["batch_xs"]),
            }
            tepres = self.sess.run(self.prediction, feed_dict=feed_dict)
            res.extend(tepres.tolist())
        return res

    def model_load(self,path):
        saver = tf.train.Saver()
        saver.restore(self.sess, path)
        return

    def model_save(self,path):
        saver = tf.train.Saver()
        saver.save(self.sess, path)
        return