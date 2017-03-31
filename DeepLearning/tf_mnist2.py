# coding: utf-8
# MNIST, higher tutorial

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gzip
import os
import tempfile

import numpy
from six.moves import urllib
from six.moves import xrange #pylint: disable=redefined-builtin
import tensorflow as tf 
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import tensorflow.examples.tutorials.mnist.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)




'''
	Test:	 0.9092
Validation:	 0.9138
[Finished in 20.4s]
	Test:	 0.9092
Validation:	 0.9138
[Finished in 4.1s]
'''
#----------------- linear regression model -------------------------

num_trn_sample, num_feature = mnist.train.images.shape
_, 				num_label 	= mnist.train.labels.shape
learning_rate = 0.01
num_iteration = 1000
num_batch = 50


#with tf.InteractiveSession() as sess:
sess = tf.InteractiveSession()
# 能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的。
x = tf.placeholder("float", shape=[None, num_feature])
y = tf.placeholder("float", shape=[None, num_label])
W = tf.Variable(tf.zeros([num_feature, num_label]))
b = tf.Variable(tf.zeros([num_label]))
sess.run(tf.initialize_all_variables())

# regression model
yest = tf.nn.softmax(tf.matmul(x,W) + b)
cross_entropy = -tf.reduce_sum(y * tf.log(yest))
# 注意，tf.reduce_sum把minibatch里的每张图片的交叉熵值都加起来了。我们计算的交叉熵是指整个minibatch的。

# train model
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
for i in range(num_iteration):
	batch = mnist.train.next_batch(num_batch)
	train_step.run(feed_dict={x:batch[0], y:batch[1]})
	# 注意，在计算图中，你可以用feed_dict来替代任何张量，并不仅限于替换占位符。

# evalute model
correct_prediction = tf.equal(tf.argmax(yest,1), tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
print("\tTest:\t", accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels}))
print("Validation:\t", accuracy.eval(feed_dict={x:mnist.validation.images, y:mnist.validation.labels}))

# test, 0.9092,  vali: 0.9138




'''
#------------------- CNN ----------------

num_trn_sample, num_feature = mnist.train.images.shape
_, 				num_label 	= mnist.train.labels.shape
learning_rate = 0.01
num_iteration = 200
num_batch = 50


def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	# tf.truncated_normal(shape, mean=0.0, stddev=1.0, dtype=tf.float32, seed=None, name=None)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

#卷积和池化  convolutional and pooling
# 在这个实例里，我们会一直使用vanilla版本。
# 我们的卷积使用1步长（stride size），0边距（padding size）的模板，保证输出和输入是同一个大小。
# 我们的池化用简单传统的2x2大小的模板做max pooling。为了代码更简洁，我们把这部分抽象成一个函数。
def conv2d(x,W):
	return tf.nn.conv2d(x,W, strides=[1,1,1,1], padding='SAME')
def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')


with tf.device('/cpu:0'):
	sess = tf.InteractiveSession()
	# 能让你在运行图的时候，插入一些计算图，这些计算图是由某些操作(operations)构成的。
	x = tf.placeholder("float", shape=[None, num_feature])
	y = tf.placeholder("float", shape=[None, num_label])



	# first convolutional layer
	W_conv1 = weight_variable([5,5,1,32])
	b_conv1 = bias_variable([32])
	x_image = tf.reshape(x, [-1,28,28,1])
	h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
	h_pool1 = max_pool_2x2(h_conv1)

	# second convolutional layer
	W_conv2 = weight_variable([5,5,32,64])
	b_conv2 = bias_variable([64])
	h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
	h_pool2 = max_pool_2x2(h_conv2)

	# 密集连接层 
	# 28-5+1=24, 24/2=12; 12-5+1=8, 8/2=4  ?????
	W_fc1 = weight_variable([7*7*64, 1024])
	b_fc1 = bias_variable([1024])
	h_pool2_flat = tf.reshape(h_pool2, [-1,7*7*64])
	h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

	# Dropout
	keep_prob = tf.placeholder("float")
	h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

	# output layer
	W_fc2 = weight_variable([1024, 10])
	b_fc2 = bias_variable([10])
	y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

	# train and evalute model
	cross_entropy = -tf.reduce_sum(y * tf.log(y_conv))
	train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
	correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y,1))
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	sess.run(tf.initialize_all_variables())
	for i in range(200): #20000
		batch = mnist.train.next_batch(50)
		if i%100 == 0:
			train_accuracy = accuracy.eval(feed_dict={x:batch[0], y:batch[1], keep_prob:1.0})
			print("Step %d, Training accuracy %g" %(i,train_accuracy))
		train_step.run(feed_dict={x:batch[0], y:batch[1], keep_prob:0.5})
	print("Test accuracy %g" %(accuracy.eval(feed_dict={x:mnist.test.images, y:mnist.test.labels, keep_prob:1.0})))
	print("Validation:   %g", accuracy.eval(feed_dict={x:mnist.validation.images, y:mnist.validation.labels, keep_prob:1.0}))

'''


