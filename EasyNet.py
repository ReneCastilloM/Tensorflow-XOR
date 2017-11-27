## A Python version of EasyNet (3 layer neural network) using TensorFlow

import numpy as np
import tensorflow as tf

Train = np.loadtxt("TrainingData.txt")
Test = np.loadtxt("TestData.txt")
NNarchitect = np.loadtxt("NeuralArchitecture.txt")

TrainInputs = Train[:,0:2]
TrainOutputs = np.transpose(np.asmatrix(Train[:,2]))
TestInputs = Test[:,0:2]
TestOutputs = np.transpose(np.asmatrix(Test[:,2]))

print (TrainInputs)
print (TrainOutputs)

N_TRAINING = len(Train)
N_TEST = len(Test)
N_INPUT_NODES = int(NNarchitect[0])
N_HIDDEN_NODES = int(NNarchitect[1])
N_OUTPUT_NODES = int(NNarchitect[2])

#print N_TRAINING
#print N_INPUT_NODES
#print N_HIDDEN_NODES
#print N_OUTPUT_NODES

x_ = tf.placeholder(tf.float32, shape=[N_TRAINING, N_INPUT_NODES], name="x-input")
y_ = tf.placeholder(tf.float32, shape=[N_TRAINING, N_OUTPUT_NODES], name="y-input")

theta1 = tf.Variable(tf.random_uniform([N_INPUT_NODES,N_HIDDEN_NODES], -1.0, 1.0), name="theta1")
theta2 = tf.Variable(tf.random_uniform([N_HIDDEN_NODES,N_OUTPUT_NODES], -1.0, 1.0), name="theta2")

bias1 = tf.Variable(tf.zeros([N_HIDDEN_NODES]), name="bias1")
bias2 = tf.Variable(tf.zeros([N_OUTPUT_NODES]), name="bias2")

layer1 = tf.sigmoid(tf.matmul(x_, theta1) + bias1)
output = tf.sigmoid(tf.matmul(layer1, theta2) + bias2)

##For the trained ANN##
x_test = tf.placeholder(tf.float32, shape=[N_TEST, N_INPUT_NODES], name="xtest-input")
y_test = tf.placeholder(tf.float32, shape=[N_TEST, N_OUTPUT_NODES], name="ytest-input")
layer1_test = tf.sigmoid(tf.matmul(x_test, theta1) + bias1)
output_test = tf.sigmoid(tf.matmul(layer1_test, theta2) + bias2)
##

## To save the trained ANN ##
tf.add_to_collection('vars', theta1)
tf.add_to_collection('vars', theta2)
tf.add_to_collection('vars', bias1)
tf.add_to_collection('vars', bias2)
#saver = tf.train.Saver()
##

cost = tf.reduce_mean(tf.square(TrainOutputs - output)) 
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cost)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for s in range(10000):
    sess.run(train_step, feed_dict={x_: TrainInputs, y_: TrainOutputs})
    if s % 1000 == 0:
    	   print('Cost ', sess.run(cost, feed_dict={x_: TrainInputs, y_: TrainOutputs}))

## Test on training data##
print('Test on training data', sess.run(output_test, feed_dict={x_test: TestInputs, y_test: TestOutputs}))

## Save the trained model ##
#saver.save(sess, 'TrainedEasyNet')

