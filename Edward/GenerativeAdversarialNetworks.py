# coding: utf-8
# http://edwardlib.org/tutorials/gan
# http://nbviewer.jupyter.org/github/blei-lab/edward/blob/master/notebooks/gan.ipynb
# 生成对抗网络 GAN    构建MNIST数字的深刻生成模型。




from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import edward as ed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import os
import tensorflow as tf

from edward.models import Uniform
from tensorflow.contrib import slim
from tensorflow.examples.tutorials.mnist import input_data




def plot(samples):
	fig = plt.figure(figsize=(4,4))
	gs = gridspec.GridSpec(4,4)
	gs.update(wspace=0.05, hspace=0.05)

	for i,sample in enumerate(samples):
		ax = plt.subplot(gs[i])
		plt.axis('off')
		ax.set_xticklabels([])
		ax.set_yticklabels([])
		ax.set_aspect('equal')
		plt.imshow(sample.reshape(28,28), cmap='Greys_r')

	return fig


ed.set_seed(42)

M = 128  # batch size during training
d = 100  # latent dimension

DATA_DIR = "../TensorFlow/MNIST_data"  #"data/mnist"
IMG_DIR = "GAN_img"  #"img"

if not os.path.exists(DATA_DIR):
	os.makedirs(DATA_DIR)
if not os.path.exists(IMG_DIR):
	os.makedirs(IMG_DIR)


mnist = input_data.read_data_sets(DATA_DIR)
x_ph = tf.placeholder(tf.float32, [M, 784])




def generative_network(eps):
	h1 = slim.fully_connected(eps, 128, activation_fn=tf.nn.relu)
	x = slim.fully_connected(h1, 784, activation_fn=tf.sigmoid)
	return x

with tf.variable_scope("Gen"):
	eps = Uniform(tf.zeros([M,d])-1.0, tf.ones([M,d]))
	x = generative_network(eps)




def discriminative_network(x):
	"""Outputs probability in logits."""
	h1 = slim.fully_connected(x, 128, activation_fn=tf.nn.relu)
	logit = slim.fully_connected(h1, 1, activation_fn=None)
	return logit


inference = ed.GANInference(data={x:x_ph}, discriminator=discriminative_network)


optimizer = tf.train.AdamOptimizer()
optimizer_d = tf.train.AdamOptimizer()

inference = ed.GANInference(data={x:x_ph}, discriminator=discriminative_network)
inference.initialize(optimizer=optimizer, optimizer_d=optimizer_d, n_iter=15000, n_print=1000)  # n_iter=15000, n_print=1000


sess = ed.get_session()
tf.global_variables_initializer().run()

idx = np.random.randint(M, size=16)
i = 0
for t in range(inference.n_iter):
	if t % inference.n_print == 0:
		samples = sess.run(x)
		samples = samples[idx,]

		fig = plot(samples)
		plt.savefig(os.path.join(IMG_DIR, '{}.png').format(str(i).zfill(3)), bbox_inches='tight')
		plt.close(fig)
		i += 1

	x_batch,_ = mnist.train.next_batch(M)
	info_dict = inference.update(feed_dict={x_ph: x_batch})
	inference.print_progress(info_dict)

'''
inference.initialize(optimizer=optimizer, optimizer_d=optimizer_d, n_iter=15, n_print=1)

Extracting ../TensorFlow/MNIST_data/train-images-idx3-ubyte.gz
Extracting ../TensorFlow/MNIST_data/train-labels-idx1-ubyte.gz
Extracting ../TensorFlow/MNIST_data/t10k-images-idx3-ubyte.gz
Extracting ../TensorFlow/MNIST_data/t10k-labels-idx1-ubyte.gz
2017-06-07 22:35:25.501223: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-07 22:35:25.502749: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-07 22:35:25.510445: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.

 1/15 [  6%] ██                             ETA: 38s | Disc Loss: 1.426 | Gen Loss: 0.746
 2/15 [ 13%] ████                           ETA: 30s | Disc Loss: 0.925 | Gen Loss: 2.547
 3/15 [ 20%] ██████                         ETA: 24s | Disc Loss: 0.694 | Gen Loss: 3.446
 4/15 [ 26%] ████████                       ETA: 22s | Disc Loss: 0.452 | Gen Loss: 3.805
 5/15 [ 33%] ██████████                     ETA: 19s | Disc Loss: 0.302 | Gen Loss: 3.866
 6/15 [ 40%] ████████████                   ETA: 17s | Disc Loss: 0.199 | Gen Loss: 3.819
 7/15 [ 46%] ██████████████                 ETA: 15s | Disc Loss: 0.141 | Gen Loss: 3.777
 8/15 [ 53%] ████████████████               ETA: 13s | Disc Loss: 0.096 | Gen Loss: 3.826
 9/15 [ 60%] ██████████████████             ETA: 11s | Disc Loss: 0.069 | Gen Loss: 3.994
10/15 [ 66%] ████████████████████           ETA: 9s | Disc Loss: 0.057 | Gen Loss: 4.169
11/15 [ 73%] ██████████████████████         ETA: 7s | Disc Loss: 0.047 | Gen Loss: 4.423
12/15 [ 80%] ████████████████████████       ETA: 5s | Disc Loss: 0.033 | Gen Loss: 4.544
13/15 [ 86%] ██████████████████████████     ETA: 3s | Disc Loss: 0.026 | Gen Loss: 4.763
14/15 [ 93%] ████████████████████████████   ETA: 1s | Disc Loss: 0.024 | Gen Loss: 4.734
15/15 [100%] ██████████████████████████████ Elapsed: 27s | Disc Loss: 0.021 | Gen Loss: 4.779
[Finished in 34.3s]
'''


'''
inference.initialize(optimizer=optimizer, optimizer_d=optimizer_d, n_iter=150, n_print=10)

Extracting ../TensorFlow/MNIST_data/train-images-idx3-ubyte.gz
Extracting ../TensorFlow/MNIST_data/train-labels-idx1-ubyte.gz
Extracting ../TensorFlow/MNIST_data/t10k-images-idx3-ubyte.gz
Extracting ../TensorFlow/MNIST_data/t10k-labels-idx1-ubyte.gz
2017-06-07 22:38:14.696075: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-07 22:38:14.705281: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-07 22:38:14.714581: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.

  1/150 [  0%]                                ETA: 467s | Disc Loss: 1.426 | Gen Loss: 0.746
 10/150 [  6%] ██                             ETA: 48s | Disc Loss: 0.057 | Gen Loss: 4.201
 20/150 [ 13%] ████                           ETA: 39s | Disc Loss: 0.017 | Gen Loss: 4.800
 30/150 [ 20%] ██████                         ETA: 35s | Disc Loss: 0.020 | Gen Loss: 4.749
 40/150 [ 26%] ████████                       ETA: 32s | Disc Loss: 0.019 | Gen Loss: 5.654
 50/150 [ 33%] ██████████                     ETA: 28s | Disc Loss: 0.039 | Gen Loss: 5.440
 60/150 [ 40%] ████████████                   ETA: 25s | Disc Loss: 0.054 | Gen Loss: 5.141
 70/150 [ 46%] ██████████████                 ETA: 22s | Disc Loss: 0.040 | Gen Loss: 5.453
 80/150 [ 53%] ████████████████               ETA: 19s | Disc Loss: 0.038 | Gen Loss: 5.343
 90/150 [ 60%] ██████████████████             ETA: 17s | Disc Loss: 0.065 | Gen Loss: 4.595
100/150 [ 66%] ████████████████████           ETA: 14s | Disc Loss: 0.095 | Gen Loss: 3.894
110/150 [ 73%] ██████████████████████         ETA: 11s | Disc Loss: 0.208 | Gen Loss: 2.820
120/150 [ 80%] ████████████████████████       ETA: 8s | Disc Loss: 0.301 | Gen Loss: 2.942
130/150 [ 86%] ██████████████████████████     ETA: 5s | Disc Loss: 0.204 | Gen Loss: 3.004
140/150 [ 93%] ████████████████████████████   ETA: 2s | Disc Loss: 0.197 | Gen Loss: 2.483
150/150 [100%] ██████████████████████████████ Elapsed: 42s | Disc Loss: 0.138 | Gen Loss: 3.263
[Finished in 46.0s]
'''


'''
inference.initialize(optimizer=optimizer, optimizer_d=optimizer_d, n_iter=1500, n_print=100)

Extracting ../TensorFlow/MNIST_data/train-images-idx3-ubyte.gz
Extracting ../TensorFlow/MNIST_data/train-labels-idx1-ubyte.gz
Extracting ../TensorFlow/MNIST_data/t10k-images-idx3-ubyte.gz
Extracting ../TensorFlow/MNIST_data/t10k-labels-idx1-ubyte.gz
2017-06-07 22:41:03.686236: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-07 22:41:03.687964: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-07 22:41:03.704307: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.

   1/1500 [  0%]                                ETA: 4035s | Disc Loss: 1.426 | Gen Loss: 0.746
 100/1500 [  6%] ██                             ETA: 101s | Disc Loss: 0.097 | Gen Loss: 3.751
 200/1500 [ 13%] ████                           ETA: 91s | Disc Loss: 0.077 | Gen Loss: 3.763
 300/1500 [ 20%] ██████                         ETA: 84s | Disc Loss: 0.032 | Gen Loss: 4.858
 400/1500 [ 26%] ████████                       ETA: 78s | Disc Loss: 0.044 | Gen Loss: 5.163
 500/1500 [ 33%] ██████████                     ETA: 72s | Disc Loss: 0.012 | Gen Loss: 6.149
 600/1500 [ 40%] ████████████                   ETA: 66s | Disc Loss: 0.008 | Gen Loss: 7.782
 700/1500 [ 46%] ██████████████                 ETA: 59s | Disc Loss: 0.005 | Gen Loss: 6.874
 800/1500 [ 53%] ████████████████               ETA: 52s | Disc Loss: 0.007 | Gen Loss: 6.468
 900/1500 [ 60%] ██████████████████             ETA: 44s | Disc Loss: 0.022 | Gen Loss: 9.796
1000/1500 [ 66%] ████████████████████           ETA: 36s | Disc Loss: 0.021 | Gen Loss: 6.107
1100/1500 [ 73%] ██████████████████████         ETA: 29s | Disc Loss: 0.015 | Gen Loss: 5.516
1200/1500 [ 80%] ████████████████████████       ETA: 22s | Disc Loss: 0.091 | Gen Loss: 6.925
1300/1500 [ 86%] ██████████████████████████     ETA: 14s | Disc Loss: 0.017 | Gen Loss: 4.847
1400/1500 [ 93%] ████████████████████████████   ETA: 7s | Disc Loss: 0.014 | Gen Loss: 6.852
1500/1500 [100%] ██████████████████████████████ Elapsed: 110s | Disc Loss: 0.034 | Gen Loss: 6.897
[Finished in 115.7s]
'''


'''
inference.initialize(optimizer=optimizer, optimizer_d=optimizer_d, n_iter=15000, n_print=1000)

Extracting ../TensorFlow/MNIST_data/train-images-idx3-ubyte.gz
Extracting ../TensorFlow/MNIST_data/train-labels-idx1-ubyte.gz
Extracting ../TensorFlow/MNIST_data/t10k-images-idx3-ubyte.gz
Extracting ../TensorFlow/MNIST_data/t10k-labels-idx1-ubyte.gz
2017-06-07 22:46:39.255736: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-07 22:46:39.258012: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-07 22:46:39.266282: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.

    1/15000 [  0%]                                ETA: 31519s | Disc Loss: 1.426 | Gen Loss: 0.746
 1000/15000 [  6%] ██                             ETA: 436s | Disc Loss: 0.021 | Gen Loss: 6.694
 2000/15000 [ 13%] ████                           ETA: 548s | Disc Loss: 0.042 | Gen Loss: 5.263
 3000/15000 [ 20%] ██████                         ETA: 621s | Disc Loss: 0.014 | Gen Loss: 5.875
 4000/15000 [ 26%] ████████                       ETA: 530s | Disc Loss: 0.043 | Gen Loss: 6.674
 5000/15000 [ 33%] ██████████                     ETA: 484s | Disc Loss: 0.062 | Gen Loss: 6.192
 6000/15000 [ 40%] ████████████                   ETA: 414s | Disc Loss: 0.213 | Gen Loss: 5.538
 7000/15000 [ 46%] ██████████████                 ETA: 349s | Disc Loss: 0.522 | Gen Loss: 3.737
 8000/15000 [ 53%] ████████████████               ETA: 293s | Disc Loss: 0.550 | Gen Loss: 3.539
 9000/15000 [ 60%] ██████████████████             ETA: 256s | Disc Loss: 0.283 | Gen Loss: 3.985
10000/15000 [ 66%] ████████████████████           ETA: 225s | Disc Loss: 0.493 | Gen Loss: 3.275
11000/15000 [ 73%] ██████████████████████         ETA: 176s | Disc Loss: 0.774 | Gen Loss: 2.250
12000/15000 [ 80%] ████████████████████████       ETA: 128s | Disc Loss: 0.649 | Gen Loss: 2.565
13000/15000 [ 86%] ██████████████████████████     ETA: 84s | Disc Loss: 0.656 | Gen Loss: 3.065
14000/15000 [ 93%] ████████████████████████████   ETA: 41s | Disc Loss: 0.556 | Gen Loss: 2.640
15000/15000 [100%] ██████████████████████████████ Elapsed: 618s | Disc Loss: 0.783 | Gen Loss: 2.133
[Finished in 621.4s]
'''



