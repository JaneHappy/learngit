# coding: utf-8
# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/tutorials/layers/cnn_mnist.py




"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 
import tensorflow as tf 

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import array_to_img, img_to_array, load_img

from tensorflow.contrib import learn
from tensorflow.contrib.learn.python.learn.estimators import model_fn as model_fn_lib

#tf.logging.set_verbosity(tf.logging.INFO)




def cnn_model_fn(features,labels,mode):
	img_width,img_height = 150,150

	"""Model function for CNN."""
	# Input layer
	# Reshape X to 4-D tensor: [batch_size, width, height, channels]
	# MNIST images are 28x28 pixels, and have one color channel
	input_layer = tf.reshape(features, [-1, img_width, img_height, 3])

	# Convolutional Layer #1
	# Computes 32 features using a 5x5 filter with ReLU activation.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 28,28,1]   ->[-1,150,150,3]
	# Output Tensor Shape: [batch_size, 28,28,32] ->[-1,150,150,96]
	conv1 = tf.layers.conv2d(inputs=input_layer, filters=32, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

	# Pooling Layer #1
	# First max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 28,28,32]
	# Output Tensor Shape: [batch_size, 14,14,32] ->[-1,50,50,96]
	pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[3,3], strides=3)

	# Convolutional Layer #2
	# Computes 64 features using a 5x5 filter.
	# Padding is added to preserve width and height.
	# Input Tensor Shape: [batch_size, 14,14,32]
	# Output Tensor Shape: [batch_size, 14,14,64] ->[-1,50,50,192]
	conv2 = tf.layers.conv2d(inputs=pool1, filters=64, kernel_size=[3,3], padding="same", activation=tf.nn.relu)

	# Pooling Layer #2
	# Second max pooling layer with a 2x2 filter and stride of 2
	# Input Tensor Shape: [batch_size, 14,14,64]
	# Output Tensor Shape: [batch_size, 7,7,64]  ->[-1,27,27,192]
	pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[3,3], strides=3)

	# Flatten tensor into a batch of vectors
	# Input Tensor Shape: [batch_size, 7,7,64]
	# Output Tensor Shape: [batch_size, 7*7*64]
	flat = tf.reshape(pool2, [-1, 27*27*64])

	# Dense Layer
	# Densely connected layer with 1024 neurons
	# Input Tensor Shape: [batch_size, 7*7*64]
	# Output Tensor Shape: [batch_size, 1024]
	dense = tf.layers.dense(inputs=flat, units=1024, activation=tf.nn.relu)

	# Add dropout operation; 0.6 probability that element will be kept
	dropout = tf.layers.dropout(inputs=dense, rate=0.5, training= mode==learn.ModeKeys.TRAIN)  #rate=0.4

	# Logits layer
	# Input Tensor Shape: [batch_size, 1024]
	# Output Tensor Shape: [batch_size, 10]
	logits = tf.layers.dense(inputs=dropout, units=10)

	loss = None
	train_op = None

	# Calculate Loss (for both TRAIN and EVAL modes)
	if mode!=learn.ModeKeys.INFER: #infer
		onehot_labels = tf.one_hot(indices=tf.cast(labels, tf.int32), depth=10)
		loss = tf.losses.softmax_cross_entropy(onehot_labels=onehot_labels, logits=logits)

	# Configure the Training Op (for TRAIN mode)
	if mode==learn.ModeKeys.TRAIN:
		train_op = tf.contrib.layers.optimize_loss(loss=loss, global_step=tf.contrib.framework.get_global_step(), learning_rate=0.001, optimizer="SGD")

	# Generate Predictions
	predictions = {
		"classes": tf.argmax(input=logits, axis=1),
		"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
	}

	# Return a ModelFnOps object
	return model_fn_lib.ModelFnOps(mode=mode, predictions=predictions, loss=loss, train_op=train_op)




def read_image():
	img_width, img_height = 150, 150
	trn_data_dir = '../../learngit_data/dogs_cats/train'
	tst_data_dir = '../../learngit_data/dogs_cats/validation'
	nb_trn_samples = 1024-64
	nb_tst_samples = 64
	epochs = 1
	batch_size = 64
	input_shape = (img_width, img_height, 3)

	#trn_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
	#tst_datagen = ImageDataGenerator(rescale=1./255)
	#trn_generator = trn_datagen.flow_from_directory(
	#	trn_data_dir, target_size=(img_width, img_height),
	#	batch_size=batch_size, class_mode='binary')
	#tst_generator = tst_datagen.flow_from_directory(
	#	tst_data_dir, target_size=(img_width, img_height),
	#	batch_size=batch_size, class_mode='binary')

	#images = [];	labels = []
	#for i in range(nb_trn_samples+nb_tst_samples):
	#	mid = (nb_trn_samples+nb_tst_samples)//2
	trn_images = [];	trn_labels = []
	tst_images = [];	tst_labels = []
	for i in range(nb_tst_samples):
		j = nb_tst_samples//2
		if i<j:
			img = load_img(tst_data_dir+'/cats/cat.'+str(i)+'.jpg')
			tst_images.append(img_to_array(img))
			tst_labels.append(0)
		else:
			img = load_img(tst_data_dir+'/dogs/dog.'+str(i-j)+'.jpg')
			tst_images.append(img_to_array(img))
			tst_labels.append(1)
	tst_images = np.array(tst_images)
	tst_labels = np.asarray(tst_labels, dtype=np.int32)
	mid = nb_tst_samples//2
	for i in range(nb_trn_samples):
		j = nb_trn_samples//2
		if i<j:
			img = load_img(trn_data_dir+'/cats/cat.'+str(mid+i)+'.jpg')
			trn_images.append(img_to_array(img))
			trn_labels.append(0)
		else:
			img = load_img(trn_data_dir+'/dogs/dog.'+str(mid+i-j)+'.jpg')
			trn_images.append(img_to_array(img))
			trn_labels.append(1)
	trn_images = np.array(trn_images)
	trn_labels = np.asarray(trn_labels, dtype=np.int32)

	return trn_images, trn_labels, tst_images, tst_labels





def main(unused_argv):
	# Load training and eval data
	'''
	mnist = learn.datasets.load_dataset("mnist")
	train_data = mnist.train.images  # Returns np.array
	train_labels = np.asarray(mnist.train.labels, dtype=np.int32)
	eval_data = mnist.test.images  # Returns np.array
	eval_labels = np.asarray(mnist.test.labels, dtype=np.int32)
	'''

	train_data,train_labels,eval_data,eval_labels = read_image()



	# Create the Estimator
	img_classifier = learn.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

	# Set up logging for predictions
	# Log the values in the "Softmax" tensor with label "probabilities"
	tensors_to_log = {"probabilities": "softmax_tensor"}
	logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

	# Train the model
	img_classifier.fit(x=train_data, y=train_labels, batch_size=64, steps=10, monitors=[logging_hook])  #steps=20000

	# Configure the accuracy metric for evaluation
	metrics = {
		"accuracy": learn.MetricSpec(
			metric_fn=tf.metrics.accuracy, prediction_key="classes"),
	}

	# Evaluate the model and print results
	eval_results = img_classifier.evaluate(x=eval_data, y=eval_labels, metrics=metrics)
	print(eval_results)




if __name__ == "__main__":
	tf.app.run()




'''
Extracting MNIST-data/train-images-idx3-ubyte.gz
Extracting MNIST-data/train-labels-idx1-ubyte.gz
Extracting MNIST-data/t10k-images-idx3-ubyte.gz
Extracting MNIST-data/t10k-labels-idx1-ubyte.gz

INFO:tensorflow:Create CheckpointSaverHook.
INFO:tensorflow:loss = 2.3137, step = 2
INFO:tensorflow:probabilities = 
INFO:tensorflow:Loss for final step: 2.3137.

INFO:tensorflow:Starting evaluation at 2017-06-16-08:22:45
INFO:tensorflow:Restoring parameters from /tmp/mnist_convnet_model/model.ckpt-2
INFO:tensorflow:Finished evaluation at 2017-06-16-08:23:23
INFO:tensorflow:Saving dict for global step 2: accuracy = 0.0604, global_step = 2, loss = 2.31161
WARNING:tensorflow:Skipping summary for global_step, must be a float or np.float32.
{'loss': 2.3116117, 'global_step': 2, 'accuracy': 0.060400002}
[Finished in 48.4s]




steps = 10
{'loss': 2.3100007, 'global_step': 12, 'accuracy': 0.064999998}
[Finished in 50.3s]
'''