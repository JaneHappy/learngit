# coding: utf-8
# References:
# http://blog.csdn.net/u013719780/article/details/53640771




#from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 




# http://blog.csdn.net/u013719780/article/details/53640771
def LinearRegression():
	# Parameters
	learning_rate = 0.01
	training_epochs = 1000 #1000
	display_step = 50

	# Generate the training data
	train_X = np.linspace(-1,1,200)
	train_Y = 2*train_X + np.random.randn(*train_X.shape)*0.2
	n_samples = train_X.shape[0]

	# tf Graph Input
	X = tf.placeholder("float")
	Y = tf.placeholder("float")

	# Initialize the variable w and b
	W = tf.Variable(np.random.randn(), name="weight")
	b = tf.Variable(np.random.randn(), name="bias")

	# Define the linear model
	pred = tf.add(tf.multiply(X, W), b)

	# Mean squared error
	cost = tf.reduce_sum(tf.pow(pred-Y, 2))/(2*n_samples)

	# Build the optimizer
	optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

	# Initializing the variables
	init = tf.initialize_all_variables()

	# Launch the graph
	with tf.Session() as sess:
		sess.run(init)

		# Fit all training data
		for epoch in range(training_epochs):
			for (x,y) in zip(train_X, train_Y):
				sess.run(optimizer, feed_dict={X:x, Y:y})
			# Display logs per epoch step
			if (epoch+1)% display_step ==0:
				c = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
				print("Epoch:", '%04d'%(epoch+1), "cost=", "{:.9f}".format(c), "W=",sess.run(W), "b=", sess.run(b))
		print("Optimization Finished!")
		training_cost = sess.run(cost, feed_dict={X:train_X, Y:train_Y})
		print("Training cost=", training_cost, "W=", sess.run(W), "b=", sess.run(b), '\n')

		# Graphic display
		plt.plot(train_X, train_Y, 'ro', label='Original data')
		plt.plot(train_X, sess.run(W)*train_X+sess.run(b), label='Fitted line')
		plt.legend()
		plt.show()

		# Generate the test data
		test_X = np.linspace(-1,1,100)
		test_Y = 2*test_X+np.random.randn(*test_X.shape)*0.2
		print("Testing... (Mean square loss Comparison)")
		testing_cost = sess.run(tf.reduce_sum(tf.pow(pred-Y, 2))/(2*test_X.shape[0]), feed_dict={X: test_X, Y:test_Y})  # same function as cost above
		print("Testing cost =", testing_cost)
		print("Absolute mean square loss difference:", abs(training_cost - testing_cost))
		plt.plot(test_X, test_Y, 'bo', label='Testing data')
		plt.plot(train_X, sess.run(W)*train_X+sess.run(b), label='Fitted line')
		plt.legend()
		plt.show()




LinearRegression()
'''
WARNING:tensorflow:From /home/ubuntu/Program/learngit/TensorFlow/algs.py:49: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
Instructions for updating:
Use `tf.global_variables_initializer` instead.
2017-06-19 23:08:34.327914: 
Epoch: 0050 cost= 0.778899372 W= 0.434037 b= 0.799931
Epoch: 0100 cost= 0.447817951 W= 0.68114 b= 0.480529
Epoch: 0150 cost= 0.283613086 W= 0.889804 b= 0.286686
Epoch: 0200 cost= 0.192970619 W= 1.06604 b= 0.169016
Epoch: 0250 cost= 0.138060316 W= 1.21491 b= 0.0975625
Epoch: 0300 cost= 0.102470256 W= 1.34069 b= 0.054153
Epoch: 0350 cost= 0.078395657 W= 1.44695 b= 0.0277637
Epoch: 0400 cost= 0.061696667 W= 1.53674 b= 0.0117069
Epoch: 0450 cost= 0.049956392 W= 1.61262 b= 0.00192499
Epoch: 0500 cost= 0.041641779 W= 1.67673 b= -0.00404439
Epoch: 0550 cost= 0.035729188 W= 1.7309 b= -0.00769574
Epoch: 0600 cost= 0.031515930 W= 1.77669 b= -0.00993641
Epoch: 0650 cost= 0.028511558 W= 1.81537 b= -0.0113174
Epoch: 0700 cost= 0.026367284 W= 1.84806 b= -0.0121736
Epoch: 0750 cost= 0.024837121 W= 1.87567 b= -0.0127086
Epoch: 0800 cost= 0.023744419 W= 1.899 b= -0.0130463
Epoch: 0850 cost= 0.022963991 W= 1.91872 b= -0.0132624
Epoch: 0900 cost= 0.022406518 W= 1.93539 b= -0.013403
Epoch: 0950 cost= 0.022008216 W= 1.94948 b= -0.0134962
Epoch: 1000 cost= 0.021723937 W= 1.96138 b= -0.0135596
Optimization Finished!
Training cost= 0.0217239 W= 1.96138 b= -0.0135596 

Testing... (Mean square loss Comparison)
Testing cost = 0.0190092
Absolute mean square loss difference: 0.0027147
[Finished in 138.0s]
'''



