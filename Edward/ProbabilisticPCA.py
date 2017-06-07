# coding: utf-8
# 概率PCA    降低潜在变量的维数。

# https://edward-cn.readthedocs.io/zh/latest/Tutorials/tutorials/#_11
# http://edwardlib.org/tutorials/probabilistic-pca
# http://nbviewer.jupyter.org/github/blei-lab/edward/blob/master/notebooks/probabilistic_pca.ipynb




from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from edward.models import Normal

plt.style.use('ggplot')




def build_toy_dataset(N, D, K, sigma=1):
	x_train = np.zeros((D,N))
	w = np.random.normal(0.0, 2.0, size=(D,K))
	z = np.random.normal(0.0, 1.0, size=(K,N))
	mean = np.dot(w,z)
	for d in range(D):
		for n in range(N):
			x_train[d,n] = np.random.normal(mean[d,n], sigma)

	print("True principal axes:")
	print(w)
	return x_train

ed.set_seed(142)

N = 5000  # number of data points
D = 2  # data dimensionality
K = 1  # latent dimensionality

x_train = build_toy_dataset(N,D,K)

# We visualize the data set.
plt.scatter(x_train[0,:], x_train[1,:], color='green', alpha=0.1)  # 'blue'
plt.axis([-10, 10, -10, 10])
plt.title("Simulated data set")
plt.show()




# Model

w = Normal(loc=tf.zeros([D,K]), scale=2.0*tf.ones([D,K]))
z = Normal(loc=tf.zeros([N,K]), scale=tf.ones([N,K]))
x = Normal(loc=tf.matmul(w,z,transpose_b=True), scale=tf.ones([D,N]))


# Inference

qw = Normal(loc=tf.Variable(tf.random_normal([D,K])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([D,K]))))
qz = Normal(loc=tf.Variable(tf.random_normal([N,K])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([N,K]))))

inference = ed.KLqp({w:qw, z:qz}, data={x:x_train})
#inference.run(n_iter=500, n_print=100, n_samples=10)
inference.run(n_iter=5, n_print=1, n_samples=10)


# Criticism

sess = ed.get_session()
print("Inferred principal axes:")
print(sess.run(qw.mean()))




# Build and then generate data from the posterior predictive distribution.

x_post = ed.copy(x, {w:qw, z:qz})
x_gen = sess.run(x_post)

plt.scatter(x_gen[0,:], x_gen[1,:], color='yellow', alpha=0.1)  #'red'
plt.axis([-10, 10, -10, 10])
plt.title("Data generated from model")
plt.show()


'''
True principal axes:
[[ 0.25947927]
 [ 1.80472372]]
2017-06-07 23:27:20.688568: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-07 23:27:20.692410: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-07 23:27:20.693669: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.

1/5 [ 20%] ██████                         ETA: 12s | Loss: 33812.719
2/5 [ 40%] ████████████                   ETA: 4s | Loss: 32534.439
3/5 [ 60%] ██████████████████             ETA: 2s | Loss: 29326.152
4/5 [ 80%] ████████████████████████       ETA: 0s | Loss: 29703.283
5/5 [100%] ██████████████████████████████ Elapsed: 3s | Loss: 26782.535
Inferred principal axes:
[[ 0.66663086]
 [-0.05523458]]
[Finished in 32.1s]
'''
