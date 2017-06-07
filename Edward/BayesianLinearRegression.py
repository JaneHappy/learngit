# coding: utf-8
# https://edward-cn.readthedocs.io/zh/latest/Tutorials/tutorials/
# 案例  贝叶斯线性回归(Bayesian linear regression)



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed 
import matplotlib.pyplot as plt 
import numpy as np 
import tensorflow as tf 

from edward.models import Normal



def build_toy_dataset(N, w):
	D = len(w)
	x = np.random.normal(0.0, 2.0, size=(N, D))
	y = np.dot(x, w) + np.random.normal(0.0, 0.01, size=N)
	return x,y

ed.set_seed(42)

N = 40  # number of data points
D = 10  # number of features

w_true = np.random.randn(D) * 0.5
X_train, y_train = build_toy_dataset(N, w_true)
X_test, y_test = build_toy_dataset(N, w_true)


# model

X = tf.placeholder(tf.float32, [N,D])
w = Normal(loc=tf.zeros(D), scale=tf.ones(D))
b = Normal(loc=tf.zeros(1), scale=tf.ones(1))
y = Normal(loc=ed.dot(X,w)+b, scale=tf.ones(N))

qw = Normal(loc=tf.Variable(tf.random_normal([D])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([D]))))
qb = Normal(loc=tf.Variable(tf.random_normal([1])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([1]))))

# 使用相对熵运行(Kullback–Leibler divergence)变分推理（这个方法在Edward中很常见），在算法中使用250次迭代和5个潜变量样本。

inference = ed.KLqp({w:qw, b:qb}, data={X:X_train, y:y_train})
inference.run(n_samples=5, n_iter=1) #n_iter=250)


# 评价与检验    评估回归的标准无非就是比较其结果对“testing”数据的预测精度。 

y_post = ed.copy(y, {w:qw, b:qb})
# y_post = Normal(ed.dot(X,qw)+qb, tf.ones(N))

print("Mean squared error on test data:")
print(ed.evaluate('mean_squared_error', data={X:X_test, y_post:y_test}))
print("Mean absolute error on test data:")
print(ed.evaluate('mean_absolute_error', data={X:X_test, y_post:y_test}))
'''
2017-06-07 20:48:53.756897: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-07 20:48:53.764590: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-07 20:48:53.766149: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
Mean squared error on test data:
18.8994
Mean absolute error on test data:
3.38316
[Finished in 7.3s]
'''



def visualise(X_data, y_data, w,b, n_samples=10):
	w_samples = w.sample(n_samples)[:,0].eval()
	b_samples = b.sample(n_samples).eval()
	plt.scatter(X_data[:,0], y_data)
	inputs = np.linspace(-8,8, num=400)
	for ns in range(n_samples):
		output = inputs * w_samples[ns] + b_samples[ns]
		plt.plot(inputs, output)
	plt.show()

# Visualize samples from the prior.
visualise(X_train, y_train, w,b)
# Visualize samples from the posterior.
visualise(X_train, y_train, qw,qb)

visualise(X_test, y_test, qw,qb)

'''
2017-06-07 20:57:21.686602: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-07 20:57:21.688559: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-07 20:57:21.700540: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
Mean squared error on test data:
18.8994
Mean absolute error on test data:
3.38316
[Finished in 80.0s]
'''


