# coding: utf-8
# https://edward-cn.readthedocs.io/zh/latest/



import numpy as np 

x_trn = np.linspace(-3, 3, num=50)
y_trn = np.cos(x_trn) + np.random.normal(0, 0.1, size=50)
x_trn = x_trn.astype(np.float32).reshape((50, 1))
y_trn = y_trn.astype(np.float32).reshape((50, 1))


import tensorflow as tf 
from edward.models import Normal 

W_0 = Normal(tf.zeros([1,2]), tf.ones([1,2])) #Normal(mu=tf.zeros([1,2]), sigma=tf.ones([1,2]))
W_1 = Normal(tf.zeros([2,1]), tf.ones([2,1]))
b_0 = Normal(tf.zeros(2), tf.ones(2))
b_1 = Normal(tf.zeros(1), tf.ones(1))

x = x_trn
y = Normal(tf.matmul(tf.tanh(tf.matmul(x,W_0)+b_0), W_1)+b_1, 0.1)

qW_0 = Normal(tf.Variable(tf.zeros([1,2])), tf.nn.softplus(tf.Variable(tf.zeros([1,2]))))
qW_1 = Normal(tf.Variable(tf.zeros([1,2])), tf.nn.softplus(tf.Variable(tf.zeros([2,1]))))
qb_0 = Normal(tf.Variable(tf.zeros(2)), tf.nn.softplus(tf.Variable(tf.zeros(2))))
qb_1 = Normal(tf.Variable(tf.zeros(1)), tf.nn.softplus(tf.Variable(tf.zeros(1))))


import edward as ed 

inference = ed.KLqp({W_0: qW_0, b_0:qb_0, W_1: qW_1, b_1: qb_1}, data={y: y_trn}) #{y: y_trn}) #
inference.run(n_iter=500)


'''
2017-06-07 17:38:54.466451: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-07 17:38:54.471479: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-07 17:38:54.478359: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
Traceback (most recent call last):
  File "/home/ubuntu/Program/learngit/Edward/basic.py", line 33, in <module>
    inference = ed.KLqp({W_0: qW_0, b_0:qb_0, W_1: qW_1, b_1: qb_1}, data={y: y_trn}) #{y: y_trn}) #
  File "/usr/local/lib/python2.7/dist-packages/edward/inferences/klqp.py", line 59, in __init__
    super(KLqp, self).__init__(*args, **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/edward/inferences/variational_inference.py", line 32, in __init__
    super(VariationalInference, self).__init__(*args, **kwargs)
  File "/usr/local/lib/python2.7/dist-packages/edward/inferences/inference.py", line 68, in __init__
    check_latent_vars(latent_vars)
  File "/usr/local/lib/python2.7/dist-packages/edward/util/random_variables.py", line 76, in check_latent_vars
    "shape: {}, {}".format(key.shape, value.shape))
TypeError: Key-value pair in latent_vars does not have same shape: (2, 1), (2, 2)
[Finished in 2.6s]
'''
