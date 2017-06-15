# coding: utf-8
# http://www.tensorfly.cn/tfdoc/get_started/introduction.html



import tensorflow as tf
import numpy as np


# 使用 NumPy 生成假数据(phony data), 总共 100 个点.

x_data = np.float32(np.random.rand(2, 100))  # 随机输入
y_data = np.dot([0.100, 0.200], x_data) + 0.300


# 构造一个线性模型

b = tf.Variable(tf.zeros([1]))
W = tf.Variable(tf.random_uniform([1,2], -1.0, 1.0))
y = tf.matmul(W, x_data) + b


# 最小化方差
loss = tf.reduce_mean(tf.square(y - y_data))
optimizer = tf.train.GradientDescentOptimizer(0.5)
train = optimizer.minimize(loss)


# 初始化变量
init = tf.initialize_all_variables()


# 启动图 (graph)
sess = tf.Session()
sess.run(init)


# 拟合平面
for step in xrange(0,201):
	sess.run(train)
	if step%20 == 0:
		print step, sess.run(W), sess.run(b)


# 得到最佳拟合结果 W: [[0.100  0.200]], b: [0.300]


'''
WARNING:tensorflow:From /home/ubuntu/Program/learngit/TensorFlow/zn_basic.py:30: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
Instructions for updating:
Use `tf.global_variables_initializer` instead.
2017-06-08 21:36:28.883634: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-08 21:36:28.888396: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-08 21:36:28.894697: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.
0 [[-0.17461336  0.4882344 ]] [ 0.87841439]
20 [[-0.00538375  0.19585729]] [ 0.36352131]
40 [[ 0.07566538  0.19396901]] [ 0.31765792]
60 [[ 0.09414081  0.19742902]] [ 0.30491543]
80 [[ 0.09853589  0.19912069]] [ 0.3013688]
100 [[ 0.09962284  0.19972518]] [ 0.30038127]
120 [[ 0.09990054  0.19991797]] [ 0.30010623]
140 [[ 0.0999733   0.19997613]] [ 0.30002961]
160 [[ 0.09999274  0.19999316]] [ 0.30000824]
180 [[ 0.099998    0.19999807]] [ 0.30000231]
200 [[ 0.09999944  0.19999945]] [ 0.30000067]
[Finished in 3.5s]
'''
