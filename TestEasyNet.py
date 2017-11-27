## A trained Python version of EasyNet (3 layer neural network) using TensorFlow

import numpy as np
import tensorflow as tf

sess = tf.Session()
new_saver = tf.train.import_meta_graph('TrainedEasyNet.meta')
new_saver.restore(sess, tf.train.latest_checkpoint('./'))
all_vars = tf.get_collection('vars')
#for v in all_vars:
    #v_ = sess.run(v)
    #print(v_)

theta1 = sess.run(all_vars[0])
theta2 = sess.run(all_vars[1])
bias1 = sess.run(all_vars[2])
bias2 = sess.run(all_vars[3])
  
print('*************************************************************')
print('*************************************************************')
print('*************************************************************')
print('Hi, welcome to EasyNet for testing.')
#print (theta1)
u = raw_input('Give an input, please:')

user = [np.float32(x) for x in u.split(',')]

print('EasyNet prediction for your input:')
x_ = np.asmatrix(user)
layer1 = tf.sigmoid(tf.matmul(x_, theta1) + bias1)
output = tf.sigmoid(tf.matmul(layer1, theta2) + bias2)
print(sess.run(output))

