# coding: utf-8
# models/tutorials/image/cifar10
# https://github.com/tensorflow/models/tree/master/tutorials/image/cifar10
'''
models/tutorials/image/cifar10
	BUILD
	README.md
	__init__.py
	cifar10.py
	cifar10_eval.py
	cifar10_input.py
	cifar10_input_test.py
	cifar10_multi_gpu_train.py
	cifar10_train.py
'''




"""Makes helper libraries available in the cifar10 package."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import re
import sys
import tarfile
from six.moves import urllib
from six.moves import xrange
import tensorflow as tf




# cifar10_input.py

# Process images of this size. Note that this differs from the orginal CIFAR image size of 32x32. If one alters this number, then the entire model architecture will change and any model would need to be retrained.
IMAGE_SIZE = 24
# Global constants describing the CIFAR-10 data set.
NUM_CLASSES = 10
NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN = 50000
NUM_EXAMPLES_PER_EPOCH_FOR_EVAL = 10000 #evaluate

def read_cifar10(filename_queue):
	"""Reads and parses examples from CIFAR10 data files."""
	class CIFAR10Record(object):
		"""docstring for CIFAR10Record"""
		pass
	result = CIFAR10Record()

	# Dimensions of the images in the CIFAR-10 dataset.
	label_bytes = 1 # 2 for CIFAR-100
	result.height = 32
	result.width = 32
	result.depth = 3
	image_bytes = result.height * result.width * result.depth
	# Every record consists of a label followed by the image, with a fixed number of bytes for each.
	record_bytes = label_bytes + image_bytes

	# Read a record, getting filenames from the filename_queue. No header or footer in the CIFAR-10 format, so we leave header_bytes and footer_bytes at their default of 0.
	reader = tf.FixedLengthRecordReader(record_bytes=record_bytes)
	result.key, value = reader.read(filename_queue) # ? result.key, result.value

	# Convert from a string to a vector of uint8 that is record_bytes long.
	record_bytes = tf.decode_raw(value, tf.uint8)

	# The first bytes represent the label, which we convert from uint8->int32.
	result.label = tf.cast(tf.strided_slice(record_bytes, [0], [label_bytes]), tf.int32)

	# The remaining bytes after the label represent the image, which we reshape from [depth*height*width] to [depth, height, width].
	depth_major = tf.reshape(
		tf.strided_slice(record_bytes, 
						 [label_bytes], 
						 [label_bytes+image_bytes]), 
		[result.depth, result.height, result.width])
	# Convert from [depth, height, width] to [height, width, depth]
	result.uint8image = tf.transpose(depth_major, [1,2,0])

	return result


def _generate_image_and_label_batch(image, label, min_queue_examples, batch_size, shuffle):
	"""Construct a queued batch of images and labels."""
	num_preprocess_threads = 16
	if shuffle:
		images, label_batch = tf.train.shuffle_batch([image,label], batch_size=batch_size, num_threads=num_preprocess_threads, capacity=min_queue_examples+3*batch_size, min_after_dequeue=min_queue_examples)
	else:
		images, label_batch = tf.train.batch([image, label], batch_size=batch_size, num_threads=num_preprocess_threads, capacity=min_queue_examples+3*batch_size)

	# Display the training images in the visualizer.
	tf.summary.image('images', images)

	return images, tf.reshape(label_batch, [batch_size])

def distorted_inputs(data_dir, batch_size):
	"""Construct distorted input for CIFAR training using the Reader ops."""
	filenames = [os.path.join(data_dir, 'data_batch_%d.bin'%i) for i in xrange(1,6)]
	for f in filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: '+f)

	# Create a queue that produces the filenames to read.
	filename_queue = tf.train.string_input_producer(filenames)

	# Read examples from files in the filename queue.
	read_input = read_cifar10(filename_queue)
	reshaped_image = tf.cast(read_input.uint8image, tf.float32)

	height = IMAGE_SIZE
	width = IMAGE_SIZE

	# Image processing for training the network. Note the many random distortions applied to the image.

	# Randomly crop a [height, width] section of the image.
	distorted_image = tf.random_crop(reshaped_image, [height,width,3])

	# Randomly flip the image horizontally.
	distorted_image = tf.image.random_flip_left_right(distorted_image)

	# Because these operations are not commutative, consider randomizing the order their operation.
	distorted_image = tf.image.random_brightness(distorted_image, max_delta=63)
	distorted_image = tf.image.random_contrast(distorted_image, lower=0.2, upper=1.8)

	# Subtract off the mean and divide by the variance of the pixels.
	float_image = tf.image.per_image_standardization(distorted_image)

	# Set the shapes of tensors.
	float_image.set_shape([height, width, 3])
	read_input.label.set_shape([1])

	# Ensure that the random shuffling has good mixing properties.
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN * min_fraction_of_examples_in_queue)
	print('Filling queue with %d CIFAR images before starting to train. This will take a few minutes.' % min_queue_examples)

	# Generate a batch of images and labels by building up a queue of examples.
	return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=True)

def inputs(eval_data, data_dir, batch_size):
	"""Construct input for CIFAR evaluation using the Reader ops."""
	if not eval_data:
		filenames = [os.path.join(data_dir, 'data_batch_%d.bin'%i) for i in xrange(1,6)]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN
	else:
		filenames = [os.path.join(data_dir, 'test_batch.bin')]
		num_examples_per_epoch = NUM_EXAMPLES_PER_EPOCH_FOR_EVAL

	for f in filenames:
		if not tf.gfile.Exists(f):
			raise ValueError('Failed to find file: ' + f)

	# Create a queue that produces the filenames to read.
	filename_queue = tf.train.string_input_producer(filenames)

	# Read examples from files in the filename queue.
	read_input = read_cifar10(filename_queue)
	reshaped_image = tf.cast(read_input.uint8image, tf.float32)

	height = IMAGE_SIZE
	width = IMAGE_SIZE

	# Image processing for evaluation.
	# Crop the central [height, width] of the image.
	resized_image = tf.image.resized_image_with_crop_or_pad(reshaped_image, height, width)

	# Subtract off the mean and divide by the variance of the pixels.
	float_image = tf.image.per_image_standardization(resized_image)

	# Set the shapes of tensors.
	float_image.set_shape([height, width, 3])
	read_input.label.set_shape([1])

	# Ensure that the random shuffling has good mixing properties.
	min_fraction_of_examples_in_queue = 0.4
	min_queue_examples = int(num_examples_per_epoch * min_fraction_of_examples_in_queue)

	# Generate a batch of images and labels by building up a queue of examples.
	return _generate_image_and_label_batch(float_image, read_input.label, min_queue_examples, batch_size, shuffle=False)




# cifar10.py

"""Builds the CIFAR-10 network."""





