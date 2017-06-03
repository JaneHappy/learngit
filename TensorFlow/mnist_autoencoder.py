# coding: utf-8
# tutorial_5_CNN.py


'''
autoencoder
	autoencoder_models
		Autoencoder.py
		DenoisingAutoencoder.py
			class AdditiveGaussianNoiseAutoencoder
			class MaskingNoiseAutoencoder
		VariationalAutoencoder.py
	AdditiveGaussianNoiseAutoencoderRunner.py
	AutoencoderRunner.py
	MaskingNoiseAutoencoderRunner.py
	VariationAutoencoderRunner.py
'''




#============================

from __future__ import print_function

import tensorflow as tf 
#import keras
import numpy as np 
import sklearn.preprocessing as prep 
from tensorflow.examples.tutorials.mnist import input_data

# from autoencoder_models.Autoencoder import Autoencoder




# Autoencoder.py

class Autoencoder(object):
	"""docstring for Antoencoder"""
	def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer()):
		#super(Antoencoder, self).__init__()
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.transfer = transfer_function
		network_weights = self._initialize_weights()
		self.weights = network_weights

		# model
		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		self.hidden = self.transfer(tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1']))
		self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
		# y1=transfer(x*w1+b1),	y2=y1*w2+b2,	cost= sigma((y2-x)^2)/2

		# cost
		self.cost = 0.5*tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0)) #?
		self.optimizer = optimizer.minimize(self.cost)

		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)

	def _initialize_weights(self):
		all_weights = dict()
		all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden], initializer=tf.contrib.layers.xavier_initializer()) #?
		all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
		all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
		all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
		return all_weights

	def transform(self, X):
		return self.sess.run(self.hidden, feed_dict={self.x: X})
	def generate(self, hidden=None):
		if hidden is None:
			hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
		return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})
	def reconstruct(self, X):
		return self.sess.run(self.reconstruction, feed_dict={self.x: X})
	# what's the meaning of self.hidden?  Answer: it's the values of hidden nodes

	def partial_fit(self, X):
		cost,opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
		return cost
	def calc_total_cost(self, X):
		return self.sess.run(self.cost, feed_dict={self.x: X})
	def getWeights(self):
		return self.sess.run(self.weights['w1'])
	def getBiases(self):
		return self.sess.run(self.weights['b1'])
	# why no ['w2,b2']?



# DenoisingAutoencoder.py

class AdditiveGaussianNoiseAutoencoder(object):
	"""docstring for AdditiveGaussianNoiseAutoencoder"""
	def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), scale=0.1):
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.transfer = transfer_function
		self.scale = tf.placeholder(tf.float32)
		self.training_scale = scale
		network_weights = self._initialize_weights()
		self.weights = network_weights

		# model
		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		self.hidden = self.transfer(tf.add(tf.matmul(self.x + scale * tf.random_normal((n_input,)), self.weights['w1']), self.weights['b1']))
		self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
		# y0=(x+scale*random) *w1+b1	## disturb x
		# y1=transfer(y0),	y2=y1*w2+b2  

		# cost
		self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
		self.optimizer = optimizer.minimize(self.cost)

		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)

	def _initialize_weights(self):
		all_weights = dict()
		all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden], initializer=tf.contrib.layers.xavier_initializer())
		all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
		all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
		all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
		return all_weights

	def transform(self, X):
		return self.sess.run(self.hidden, feed_dict={self.x: X, self.scale: self.training_scale})
	def generate(self, hidden=None):
		if hidden is None:
			hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
		return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})
	def reconstruct(self, X):
		return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.scale: self.training_scale})

	def partial_fit(self, X):
		cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.scale: self.training_scale})
		return cost
	def calc_total_cost(self, X):
		return self.sess.run(self.cost, feed_dict={self.x: X, self.scale:self.training_scale})
	def getWeights(self):
		return self.sess.run(self.weights['w1'])
	def getBiases(self):
		return self.sess.run(self.weights['b1'])

class MaskingNoiseAutoencoder(object):
	"""docstring for MaskingNoiseAutoencoder"""
	def __init__(self, n_input, n_hidden, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(), dropout_probability=0.95):
		self.n_input = n_input
		self.n_hidden = n_hidden
		self.tranfer = transfer_function
		self.dropout_probability = dropout_probability
		self.keep_prob = tf.placeholder(tf.float32)
		network_weights = self._initialize_weights()
		self.weights = network_weights

		# model
		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		self.hidden = self.tranfer(tf.add(tf.matmul(tf.nn.dropout(self.x, self.keep_prob), self.weights['w1']), self.weights['b1']))
		self.reconstruction = tf.add(tf.matmul(self.hidden, self.weights['w2']), self.weights['b2'])
		# y0=dropout(x,?) *w1+b1	## mask x
		# y1=tranfer(y0),	y2=y1*w2+b2

		# cost
		self.cost = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
		self.optimizer = optimizer.minimize(self.cost)

		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)

	def _initialize_weights(self):
		all_weights = dict()
		all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden], initializer=tf.contrib.layers.xavier_initializer())
		all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype = tf.float32))
		all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype = tf.float32))
		all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype = tf.float32))
		return all_weights

	def transform(self, X):
		return self.sess.run(self.hidden, feed_dict={self.x: X, self.keep_prob: 1.0})
	def generate(self, hidden=None):
		if hidden is None:
			hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
		return self.sess.run(self.reconstruction, feed_dict={self.hidden: hidden})
	def reconstruct(self, X):
		return self.sess.run(self.reconstruction, feed_dict={self.x: X, self.keep_prob: 1.0})

	def partial_fit(self, X):
		cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X, self.keep_prob: self.dropout_probability})
		return cost
	def calc_total_cost(self, X):
		return self.sess.run(self.cost, feed_dict={self.x: X, self.keep_prob: 1.0})
	def getWeights(self):
		return self.sess.run(self.weights['w1'])
	def getBiases(self):
		return self.sess.run(self.weights['b1'])



# VariationalAutoencoder.py

class VariationalAutoencoder(object):
	"""docstring for VariationalAutoencoder"""
	def __init__(self, n_input, n_hidden, optimizer=tf.train.AdamOptimizer()):
		self.n_input = n_input
		self.n_hidden = n_hidden
		network_weights = self._initialize_weights()
		self.weights = network_weights

		# model
		self.x = tf.placeholder(tf.float32, [None, self.n_input])
		self.z_mean = tf.add(tf.matmul(self.x, self.weights['w1']), self.weights['b1'])
		self.z_log_sigma_sq = tf.add(tf.matmul(self.x, self.weights['log_sigma_w1']), self.weights['log_sigma_b1'])
		# zbar=x*w1+b1,	zlog_s2=x*w1_log+b1_log

		# sample from gaussian distribution
		eps = tf.random_normal(tf.stack([tf.shape(self.x)[0], self.n_hidden]), 0, 1, dtype=tf.float32)
		self.z = tf.add(self.z_mean, tf.multiply(tf.sqrt(tf.exp(self.z_log_sigma_sq)), eps))
		# eps=rondom([x.shape[0],n_hidden], mean,stddev) 	##stddev is sqrt(delta)
		# z= zbar+ sqrt(zlog_s2).*eps

		self.reconstruction = tf.add(tf.matmul(self.z, self.weights['w2']), self.weights['b2'])
		# y = z*w2+b2

		# cost
		reconstr_loss = 0.5 * tf.reduce_sum(tf.pow(tf.subtract(self.reconstruction, self.x), 2.0))
		latent_loss = -0.5 * tf.reduce_sum(1 + self.z_log_sigma_sq - tf.square(self.z_mean) - tf.exp(self.z_log_sigma_sq), axis=1) # by row
		self.cost = tf.reduce_mean(reconstr_loss + latent_loss)
		self.optimizer = optimizer.minimize(self.cost)
		# reloss=sumall((y-x).^2)/2,	laloss=-0.5*sum(1+zlog-zbar.^2-e.^(zlog))

		init = tf.global_variables_initializer()
		self.sess = tf.Session()
		self.sess.run(init)

	def _initialize_weights(self):
		all_weights = dict()
		all_weights['w1'] = tf.get_variable("w1", shape=[self.n_input, self.n_hidden], initializer=tf.contrib.layers.xavier_initializer())
		all_weights['log_sigma_w1'] = tf.get_variable("log_sigma_w1", shape=[self.n_input, self.n_hidden], initializer=tf.contrib.layers.xavier_initializer())
		all_weights['b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
		all_weights['log_sigma_b1'] = tf.Variable(tf.zeros([self.n_hidden], dtype=tf.float32))
		all_weights['w2'] = tf.Variable(tf.zeros([self.n_hidden, self.n_input], dtype=tf.float32))
		all_weights['b2'] = tf.Variable(tf.zeros([self.n_input], dtype=tf.float32))
		return all_weights

	def transform(self, X):
		return self.sess.run(self.z_mean, feed_dict={self.x: X})
	def generate(self, hidden=None):
		if hidden is None:
			hidden = self.sess.run(tf.random_normal([1, self.n_hidden]))
		return self.sess.run(self.reconstruction, feed_dict={self.z: hidden})
	def reconstruct(self, X):
		return self.sess.run(self.reconstruction, feed_dict={self.x: X})

	def partial_fit(self, X):
		cost, opt = self.sess.run((self.cost, self.optimizer), feed_dict={self.x: X})
		return cost
	def calc_total_cost(self, X):
		return self.sess.run(self.cost, feed_dict={self.x: X})
	def getWeights(self):
		return self.sess.run(self.weights['w1'])
	def getBiases(self):
		return self.sess.run(self.weights['b1'])











# (X_trn,y_trn),(X_tst,y_tst) = keras.datasets.mnist.load_data()


# AutoencoderRunner.py

def AutoencoderRunner():
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	def standard_scale(X_train, X_test):
		preprocessor = prep.StandardScaler().fit(X_train)
		X_train = preprocessor.transform(X_train)
		X_test = preprocessor.transform(X_test)
		return X_train, X_test

	def get_random_block_from_data(data, batch_size):
		start_index = np.random.randint(0, len(data)-batch_size)
		return data[start_index:(start_index+batch_size)]

	X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

	n_samples = int(mnist.train.num_examples)
	training_epochs = 1 #20
	batch_size = 128
	display_step = 1

	autoencoder = Autoencoder(n_input=784, n_hidden=200, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(learning_rate=0.001))

	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(n_samples / batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_xs = get_random_block_from_data(X_train, batch_size)
			# Fit training using batch data
			cost = autoencoder.partial_fit(batch_xs)
			# Compute average loss
			avg_cost += cost/n_samples*batch_size
		# Display logs per epoch step 
		if epoch%display_step==0:
			print("Epoch:", '%04d'%(epoch+1), "cost=","{:.9f}".format(avg_cost))
	print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))



'''
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

def standard_scale(X_train, X_test):
	preprocessor = prep.StandardScaler().fit(X_train)
	X_train = preprocessor.transform(X_train)
	X_test = preprocessor.transform(X_test)
	return X_train, X_test

def get_random_block_from_data(data, batch_size):
	start_index = np.random.randint(0, len(data)-batch_size)
	return data[start_index:(start_index+batch_size)]

X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

n_samples = int(mnist.train.num_examples)
training_epochs = 1 #20
batch_size = 128
display_step = 1

autoencoder = Autoencoder(n_input=784, n_hidden=200, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(learning_rate=0.001))

for epoch in range(training_epochs):
	avg_cost = 0.
	total_batch = int(n_samples / batch_size)
	# Loop over all batches
	for i in range(total_batch):
		batch_xs = get_random_block_from_data(X_train, batch_size)
		# Fit training using batch data
		cost = autoencoder.partial_fit(batch_xs)
		# Compute average loss
		avg_cost += cost/n_samples*batch_size
	# Display logs per epoch step 
	if epoch%display_step==0:
		print("Epoch:", '%04d'%(epoch+1), "cost=","{:.9f}".format(avg_cost))
print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))


Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
Epoch: 0001 cost= 19218.661909091
Total cost: 1.13131e+06
[Finished in 27.9s]
'''



# AdditiveGaussianNoiseAutoencoderRunner.py

def AdditiveGaussianNoiseAutoencoderRunner():
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	def standard_scale(X_train, X_test):
		preprocessor = prep.StandardScaler().fit(X_train)
		X_train = preprocessor.transform(X_train)
		X_test = preprocessor.transform(X_test)
		return X_train, X_test

	def get_random_block_from_data(data, batch_size):
		start_index = np.random.randint(0, len(data) - batch_size)
		return data[start_index:(start_index + batch_size)]

	X_train,X_test = standard_scale(mnist.train.images, mnist.test.images)

	n_samples = int(mnist.train.num_examples)
	training_epochs = 1 #20
	batch_size = 128
	display_step = 1

	autoencoder = AdditiveGaussianNoiseAutoencoder(n_input=784, n_hidden=200, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(learning_rate=0.001), scale=0.01)

	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(n_samples / batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_xs = get_random_block_from_data(X_train, batch_size)
			# Fit training using batch data
			cost = autoencoder.partial_fit(batch_xs)
			# compute average loss
			avg_cost += cost/n_samples*batch_size
		# Display logs per epoch step
		if epoch%display_step==0:
			print("Epoch:", '%04d'%(epoch+1), "cost=","{:.9f}".format(avg_cost))
	print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))




# MaskingNoiseAutoencoderRunner.py

def MaskingNoiseAutoencoderRunner():
	mnist = input_data.read_data_sets('MNIST_data', one_hot = True)

	def standard_scale(X_train, X_test):
		preprocessor = prep.StandardScaler().fit(X_train)
		X_train = preprocessor.transform(X_train)
		X_test = preprocessor.transform(X_test)
		return X_train, X_test

	def get_random_block_from_data(data, batch_size):
		start_index = np.random.randint(0, len(data) - batch_size)
		return data[start_index:(start_index + batch_size)]

	X_train, X_test = standard_scale(mnist.train.images, mnist.test.images)

	n_samples = int(mnist.train.num_examples)
	training_epochs = 1 #100
	batch_size = 128
	display_step = 1

	autoencoder = MaskingNoiseAutoencoder(n_input=784, n_hidden=200, transfer_function=tf.nn.softplus, optimizer=tf.train.AdamOptimizer(learning_rate=0.001), dropout_probability=0.95)

	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(n_samples / batch_size)
		for i in range(total_batch):
			batch_xs = get_random_block_from_data(X_train, batch_size)
			cost = autoencoder.partial_fit(batch_xs)
			avg_cost += cost/n_samples*batch_size
		if epoch%display_step==0:
			print("Epoch:", '%04d'%(epoch+1), "cost=","{:.9}".format(avg_cost))
	print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))




# VariationAutoencoderRunner.py

def VariationalAutoencoderRunner():
	mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

	def min_max_scale(X_train, X_test):
		preprocessor = prep.MinMaxScaler().fit(X_train)
		X_train = preprocessor.transform(X_train)
		X_test = preprocessor.transform(X_test)
		return X_train,X_test

	def get_random_block_from_data(data, batch_size):
		start_index = np.random.randint(0, len(data)-batch_size)
		return data[start_index:(start_index+batch_size)]

	X_train,X_test = min_max_scale(mnist.train.images, mnist.test.images)
	n_samples = int(mnist.train.num_examples)
	training_epochs = 1 #20
	batch_size = 128
	display_step = 1

	autoencoder = VariationalAutoencoder(n_input=784, n_hidden=200, optimizer=tf.train.AdamOptimizer(learning_rate=0.001))

	for epoch in range(training_epochs):
		avg_cost = 0.
		total_batch = int(n_samples/batch_size)
		# Loop over all batches
		for i in range(total_batch):
			batch_xs = get_random_block_from_data(X_train, batch_size)
			# Fit training using batch data
			cost = autoencoder.partial_fit(batch_xs)
			# Compute average loss
			avg_cost += cost/n_samples*batch_size
		# Display logs per epoch step
		if epoch%display_step==0:
			print("Epoch:", '%04d'%(epoch+1), "cost=","{:.9f}".format(avg_cost))
	print("Total cost: " + str(autoencoder.calc_total_cost(X_test)))






# ------------------------

# AutoencoderRunner()

#AdditiveGaussianNoiseAutoencoderRunner()
'''
Extracting MNIST_data/train-images-idx3-ubyte.gz
Extracting MNIST_data/train-labels-idx1-ubyte.gz
Extracting MNIST_data/t10k-images-idx3-ubyte.gz
Extracting MNIST_data/t10k-labels-idx1-ubyte.gz
2017-06-02 01:44:20.689328: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
Epoch: 0001 cost= 18688.993296591
Total cost: 1.08278e+06
[Finished in 34.3s]
'''

#MaskingNoiseAutoencoderRunner()
'''
2017-06-02 02:01:05.431119: 
Epoch: 0001 cost= 19225.5329
Total cost: 1.14209e+06
[Finished in 33.5s]
'''

#VariationalAutoencoderRunner()
'''
2017-06-03 12:53:22.389628: 
Epoch: 0001 cost= 1148.902823722
Total cost: 34770.7
[Finished in 38.1s]
'''




