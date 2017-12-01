import numpy as np
import tensorflow as tf  #TF 1.1.0rc1
tf.logging.set_verbosity(tf.logging.ERROR)
import matplotlib.pyplot as plt
from model import Model,sample_batch
from preprocess import preprocess

def main():
  
  [X_train, y_train, X_test, y_test] = preprocess()
  N,sl = X_train.shape
  num_classes = len(np.unique(y_train))

  """Hyperparamaters"""
  batch_size = 100
  max_iterations = N * 2 * batch_size
  dropout = 0.8
  config = {    'num_layers' :    3,               #number of layers of stacked RNN's
                'hidden_size' :   128,             #memory cells in a layer
                'max_grad_norm' : 5,             
                'batch_size' :    batch_size,
                'learning_rate' : .005,
                'sl':             sl,
                'num_classes':    num_classes}

  epochs = np.floor(batch_size*max_iterations / N)
  print('Train %.0f samples in approximately 2 epochs' %(N))

  model = Model(config)
  sess = tf.Session() 
  sess.run(model.init_op)

  cost_train_ma = -np.log(1/float(num_classes)+1e-9)  #Moving average training cost
  acc_train_ma = 0.0
  for i in range(max_iterations):
    X_batch, y_batch = sample_batch(X_train,y_train,batch_size)
    cost_train, acc_train, _ = sess.run([model.cost,model.accuracy, model.train_op],
    feed_dict = {model.input: X_batch,model.labels: y_batch,model.keep_prob:dropout})
    cost_train_ma = cost_train_ma*0.99 + cost_train*0.01
    acc_train_ma = acc_train_ma*0.99 + acc_train*0.01
  
  #Evaluate validation performance
    if i%100 == 0:
      X_batch, y_batch = sample_batch(X_test,y_test,batch_size)
      cost_val, summ, acc_val = sess.run([model.cost,model.merged,model.accuracy],feed_dict = {model.input: X_batch, model.labels: y_batch, model.keep_prob:1.0})
      print('At %5.0f/%5.0f: COST %5.3f/%5.3f(%5.3f) -- Acc %5.3f/%5.3f(%5.3f)' %(i,max_iterations,cost_train,cost_val,cost_train_ma,acc_train,acc_val,acc_train_ma))
      plt.plot(i,acc_train,'r*')
      plt.plot(i,cost_train,'kd')
    
  cost_val, summ, acc_val = sess.run([model.cost,model.merged,model.accuracy],feed_dict = {model.input: X_test, model.labels: y_test, model.keep_prob:1.0})
  epoch = float(i)*batch_size/N
  print('Trained %.1f epochs, accuracy is %5.3f and cost is %5.3f'%(epoch,acc_val,cost_val))
  plt.xlabel('Iteration')
  plt.ylabel('Accuracy & Cost')
  plt.show()
  
 
if __name__ == '__main__':
  main()



