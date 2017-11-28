"""
LSTM for time series classification

This model takes in time series and class labels.
The LSTM models the time series. A fully-connected layer
generates an output to be classified with Softmax
"""

import numpy as np
import tensorflow as tf  #TF 1.1.0rc1
tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib.pyplot as plt
from tsc_model import Model,sample_batch,load_data
from sklearn import preprocessing
from sklearn.cluster import KMeans


#Set these directories
# direc = '/home/rob/Dropbox/ml_projects/LSTM/UCR_TS_Archive_2015'

def data_preprocessing(file):
    data, tmp, pre, tmp2, data_raw = {}, [], '', [], {}
    for line in open(file,'r').readlines():
        line = line.split(',')
        if line[0] == 'name' and pre != '':
            tmp2 = tmp
            data_raw[pre] = np.asarray(tmp2,dtype='f')
            data[pre] = preprocessing.normalize(np.asarray(tmp2,dtype='f'))
            tmp = []
        elif line[0] == 'name': 
            data['name'] = line[2:]
            data_raw['name'] = line[2:]
        if line[2].isdigit() == True:
            row = []
            for i in range(2,47):
                row.append(0.0 if line[i]=='' else float(line[i]))  
            tmp.append(np.asarray(row, dtype='f'))
            pre = line[0]
    tmp2 = tmp
    data[pre] = preprocessing.normalize(np.asarray(tmp2,dtype='f'))
    data_raw[pre] = np.asarray(tmp2,dtype='f')
    return [data, data_raw]

def example0():
  train_pos_1 = 'G_Fifty_1'
  train_neg_1 = 'G_Hundred_1'
  train_pos_2 = 'G_Fifty_2'
  train_neg_2 = 'G_Hundred_2'
  train_pos_3 = 'G_Fifty_3'
  train_neg_3 = 'G_Hundred_3'
  train_pos_4 = 'G_Fifty_4'
  train_neg_4 = 'G_Hundred_4'
  train_pos_5 = 'G_Fifty_5'
  train_neg_5 = 'G_Hundred_5'
  train_pos_6 = 'G_Fifty_6'
  train_neg_6 = 'G_Hundred_6'
  train_pos_7 = 'G_Fifty_7'
  train_neg_7 = 'G_Hundred_7'
  train_pos_8 = 'G_Fifty_8'
  train_neg_8 = 'G_Hundred_8'


  test_pos = 'G_Fifty_9'
  test_neg = 'G_Hundred_9'
  data,raw = data_preprocessing('data.csv')

  x1 = raw[train_pos_1]
  x2 = raw[train_neg_1]
  x3 = raw[train_pos_2]
  x4 = raw[train_neg_2]
  x5 = raw[train_pos_3]
  x6 = raw[train_neg_3]
  x7 = raw[train_pos_4]
  x8 = raw[train_neg_4]
  x9 = raw[train_pos_5]
  x10 = raw[train_neg_5]
  x11 = raw[train_pos_6]
  x12 = raw[train_neg_6]
  x13 = raw[train_pos_7]
  x14 = raw[train_neg_7]
  x15 = raw[train_pos_8]
  x16 = raw[train_neg_8]

  x1_ = raw[test_pos]
  x2_ = raw[test_neg]
  # X1 = np.concatenate((x1[:,4],x2[:,4],x3[:,4],x4[:,4],x5[:,4],x6[:,4],x7[:,4],x8[:,4],
  #                      x9[:,4],x10[:,4],x11[:,4],x12[:,4],x13[:,4],x14[:,4],x15[:,4],x16[:,4] ), axis=0).reshape(-1,1)
  # X2 = np.concatenate((x1[:,26],x2[:,26],x3[:,26],x4[:,26],x5[:,26],x6[:,26],x7[:,26],x8[:,26],
  #                     x9[:,26],x10[:,26],x11[:,26],x12[:,26],x13[:,26],x14[:,26],x15[:,26],x16[:,26] ), axis=0).reshape(-1,1)
  X = np.concatenate((x1,x2,x3,x4,x5,x6,x7,x8,
                       x9,x10,x11,x12,x13,x14,x15,x16 ), axis=0)
  # X  =  np.concatenate((X1, X2), axis=1)
  y  = np.concatenate((0*np.ones(len(x1)),np.ones(len(x2)),0*np.ones(len(x3)),np.ones(len(x4)), 
                       0*np.ones(len(x5)),np.ones(len(x6)),0*np.ones(len(x7)),np.ones(len(x8)),
                       0*np.ones(len(x9)),np.ones(len(x10)),0*np.ones(len(x11)),np.ones(len(x12)),
                       0*np.ones(len(x13)),np.ones(len(x14)),0*np.ones(len(x15)),np.ones(len(x16))),axis=0)

  # X1_ = np.concatenate((x1_[:,4],x2_[:,4]), axis=0).reshape(-1,1)
  # X2_ = np.concatenate((x1_[:,26],x2_[:,26]), axis=0).reshape(-1,1)
  # X_ =  np.concatenate((X1_, X2_), axis=1)
  X_ = np.concatenate((x1_,x2_), axis=0)
  y_ = np.concatenate((0*np.ones(len(x1_)),np.ones(len(x2_))),axis=0)

  return [X, y, X_, y_]



def main():
  """Load the data"""
  ratio = np.array([0.8,0.9]) #Ratios where to split the training and validation set
  #X_train,X_val,X_test,y_train,y_val,y_test = load_data(direc,ratio,dataset='ChlorineConcentration')
  [X, y, X_, y_] = example0()
  X_train,X_val,X_test,y_train,y_val,y_test = X, X, X_, y, y, y_

  print X_test.shape
  print y_test.shape
  N,sl = X_train.shape
  num_classes = len(np.unique(y_train))

  """Hyperparamaters"""
  batch_size = 100
  max_iterations = 5000
  dropout = 0.8
  config = {    'num_layers' :    3,               #number of layers of stacked RNN's
                'hidden_size' :   128,             #memory cells in a layer
                'max_grad_norm' : 5,             #maximum gradient norm during training
                'batch_size' :    batch_size,
                'learning_rate' : .005,
                'sl':             sl,
                'num_classes':    num_classes}

  epochs = np.floor(batch_size*max_iterations / N)
  print('Train %.0f samples in approximately %d epochs' %(N,epochs))

  #Instantiate a model
  model = Model(config)

  """Session time"""
  sess = tf.Session() #Depending on your use, do not forget to close the session
  sess.run(model.init_op)

  cost_train_ma = -np.log(1/float(num_classes)+1e-9)  #Moving average training cost
  acc_train_ma = 0.0
  try:
    for i in range(max_iterations):
      X_batch, y_batch = sample_batch(X_train,y_train,batch_size)

      #Next line does the actual training
      cost_train, acc_train, _ = sess.run([model.cost,model.accuracy, model.train_op],
        feed_dict = {model.input: X_batch,model.labels: y_batch,model.keep_prob:dropout})
      cost_train_ma = cost_train_ma*0.99 + cost_train*0.01
      acc_train_ma = acc_train_ma*0.99 + acc_train*0.01
      if i%100 == 0:
      #Evaluate validation performance
        X_batch, y_batch = sample_batch(X_val,y_val,batch_size)
        cost_val, summ, acc_val = sess.run([model.cost,model.merged,model.accuracy],feed_dict = {model.input: X_batch, model.labels: y_batch, model.keep_prob:1.0})
        print('At %5.0f/%5.0f: COST %5.3f/%5.3f(%5.3f) -- Acc %5.3f/%5.3f(%5.3f)' %(i,max_iterations,cost_train,cost_val,cost_train_ma,acc_train,acc_val,acc_train_ma))
        plt.plot(i,acc_train,'r*')
        plt.plot(i,cost_train,'kd')
    
  except KeyboardInterrupt:
    pass
  cost_val, summ, acc_val = sess.run([model.cost,model.merged,model.accuracy],feed_dict = {model.input: X_test, model.labels: y_test, model.keep_prob:1.0})
  epoch = float(i)*batch_size/N
  print('Trained %.1f epochs, accuracy is %5.3f and cost is %5.3f'%(epoch,acc_val,cost_val))
  plt.xlabel('Iteration')
  plt.ylabel('Accuracy & Cost')
  plt.show()
  
 

if __name__ == '__main__':
  main()
#now run in your terminal:
# $ tensorboard --logdir = <summaries_dir>
# Replace <summaries_dir> with your own dir



