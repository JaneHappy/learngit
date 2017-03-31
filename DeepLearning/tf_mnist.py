# coding: utf-8
# MNIST


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

# implement regression model
num_trn_sample,num_feat = mnist.train.images.shape
_, num_label 			= mnist.train.labels.shape #feature
learning_rate = 0.01
iterations = 1000
num_batch = 100

x = tf.placeholder(tf.float32, [None,num_feat])
y = tf.placeholder(tf.float32, [None,num_label])
W = tf.Variable(tf.zeros([num_feat, num_label]))
b = tf.Variable(tf.zeros([num_label]))
yest = tf.nn.softmax(tf.matmul(x,W) + b) #y_estimate
cross_entropy = -tf.reduce_sum(y*tf.log(yest))

train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)
init = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init)
	for i in range(iterations):
		batch_xs, batch_ys = mnist.train.next_batch(num_batch)
		sess.run(train_step, feed_dict={x:batch_xs, y:batch_ys})
	correct_prediction = tf.equal(tf.argmax(yest,1), tf.argmax(y,1)) #?? why 1
	accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
	print("Test:\t\t", sess.run(accuracy, feed_dict={x:mnist.test.images, y:mnist.test.labels}))
	print("Validation:\t", sess.run(accuracy, feed_dict={x:mnist.validation.images, y:mnist.validation.labels}))

'''
Test:		 0.9185
Validation:	 0.9226
[Finished in 4.8s]
'''



