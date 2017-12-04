import numpy as np
import tensorflow as tf
from random import randint
import datetime
# from preprocess import preprocess

# [X_train, y_train, X_test, y_test] = preprocess()
X_train = np.load('train_data.npy', encoding = 'latin1')
y_train = np.load('train_label.npy', encoding = 'latin1')
X_test = np.load('test_data.npy', encoding = 'latin1')
y_test = np.load('test_label.npy', encoding = 'latin1')

batchSize = 24
lstmUnits = 64
numClasses = 2
iterations = 100000
maxSeqLength = 250 #Maximum length of sentence
numDimensions = 100 #Dimensions for each word vector

def getTrainBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength, numDimensions])
    for i in range(batchSize):
        if (i % 2 == 0): 
            num = randint(0,12499)
            labels.append([0,1])
        else:
            num = randint(12500,24999)
            labels.append([1,0])
        ml = min(len(X_train[num]), maxSeqLength)
        arr[i][:ml] = X_train[num][:ml]
    return arr, labels

def getTestBatch():
    labels = []
    arr = np.zeros([batchSize, maxSeqLength])
    for i in range(batchSize):
        if (i % 2 == 0): 
            num = randint(0,12499)
            labels.append([0,1])
        else:
            num = randint(12500,24999)
            labels.append([1,0])
        ml = min(len(X_test[num]), maxSeqLength)
        arr[i][:ml] = X_test[num][:ml]
    return arr, labels

tf.reset_default_graph()

labels = tf.placeholder(tf.float32, [batchSize, numClasses])
input_data = tf.placeholder(tf.float32, [batchSize, maxSeqLength, numDimensions])

data = tf.Variable(tf.zeros([batchSize, maxSeqLength, numDimensions]),dtype=tf.float32)
data = tf.assign(data, input_data)
# data = tf.nn.embedding_lookup(wordVectors,input_data)

lstmCell = tf.contrib.rnn.BasicLSTMCell(lstmUnits)
lstmCell = tf.contrib.rnn.DropoutWrapper(cell=lstmCell, output_keep_prob=0.75)
value, _ = tf.nn.dynamic_rnn(lstmCell, data, dtype=tf.float32)

weight = tf.Variable(tf.truncated_normal([lstmUnits, numClasses]))
bias = tf.Variable(tf.constant(0.1, shape=[numClasses]))
value = tf.transpose(value, [1, 0, 2])
last = tf.gather(value, int(value.get_shape()[0]) - 1)
prediction = (tf.matmul(last, weight) + bias)

correctPred = tf.equal(tf.argmax(prediction,1), tf.argmax(labels,1))
accuracy = tf.reduce_mean(tf.cast(correctPred, tf.float32))

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=labels))
optimizer = tf.train.AdamOptimizer().minimize(loss)

# Train process
sess = tf.InteractiveSession()
saver = tf.train.Saver()
sess.run(tf.global_variables_initializer())


tf.summary.scalar('Loss', loss)
tf.summary.scalar('Accuracy', accuracy)
merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir, sess.graph)

for i in range(iterations):
   #Next Batch of reviews
   nextBatch, nextBatchLabels = getTrainBatch();
   sess.run(optimizer, {input_data: nextBatch, labels: nextBatchLabels})
   
   #Write summary to Tensorboard
   if (i % 50 == 0):
       summary = sess.run(merged, {input_data: nextBatch, labels: nextBatchLabels})
       writer.add_summary(summary, i)

   # Save the network every 10,000 training iterations
   if (i % 10000 == 0 and i != 0):
       save_path = saver.save(sess, "models/pretrained_lstm.ckpt", global_step=i)
       print("saved to %s" % save_path)
writer.close()

