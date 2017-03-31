# coding: utf-8
# 曼德布洛特(Mandelbrot)集合


# 导入仿真库
import tensorflow as tf
import numpy as np 
import math
# 导入可视化库
import PIL.Image as plt
from cStringIO import StringIO 
from IPython.display import clear_output, Image, display
import scipy.ndimage as nd 

# 现在我们将定义一个函数来显示迭代计算出的图像。
def DisplayFractal(a, fmt='jpeg'):
	"""显示迭代计算出的彩色分形图像。"""
	a_cyclic = (6.28*a/20.0).reshape(list(a.shape)+[1])
	img = np.concatenate([10+20*np.cos(a_cyclic), 30+50*np.sin(a_cyclic), 155-80*np.cos(a_cyclic)], 2)
	img[a==a.max()] = 0
	a = img
	a = np.uint8(np.clip(a, 0, 255))
	f = StringIO()
	plt.fromarray(a).save(f, fmt)
	display(Image(data=f.getvalue()))

#会话（session）和变量（variable）初始化
#为了操作的方便，我们常常使用交互式会话（interactive session），但普通会话（regular session）也能正常使用。
sess = tf.InteractiveSession()
# 使用NumPy创建一个在[-2,2]x[-2,2]范围内的2维复数数组
Y,X = np.mgrid[-1.3:1.3:0.005, -2:1:0.005]
Z = X+1j*Y
# 定义并初始化一组TensorFlow的张量 （tensors）。
xs = tf.constant(Z.astype("complex64"))
zs = tf.Variable(xs)
ns = tf.Variable(tf.zeros_like(xs, "float32"))
# TensorFlow在使用之前需要你明确给定变量的初始值。
tf.initialize_all_variables().run()

#定义并运行计算
#现在我们指定更多的计算...
# 计算一个新值z: z^2 + x
zs_ = zs*zs + xs
# 这个新值会发散吗？
#not_diverged = tf.complex_abs(zs_) < 4
def complex_abs_sq(z):
	ans = tf.real(z)**2 + tf.imag(z)**2
	return ans**(1.0/2.0) #tf.cast(ans, "float")
	#return tf.real(z)*tf.real(z) + tf.imag(z)*tf.imag(z)
not_diverged = complex_abs_sq(zs_)<4 #math.sqrt(complex_abs_sq(zs_)) < 4
# 更新zs并且迭代计算。
# 说明：在这些值发散之后，我们仍然在计算zs，这个计算消耗特别大！
#      如果稍微简单点，这里有更好的方法来处理。
step = tf.group(zs.assign(zs_), ns.assign_add(tf.cast(not_diverged, "float32")))
# ...继续执行几百个步骤
for i in range(200):	step.run()
# 让我们看看我们得到了什么。
DisplayFractal(ns.eval())


'''
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE3 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
WARNING:tensorflow:From /home/dell/Programming/tf_MandelbrotSet.py:38: initialize_all_variables (from tensorflow.python.ops.variables) is deprecated and will be removed after 2017-03-02.
Instructions for updating:
Use `tf.global_variables_initializer` instead.
<IPython.core.display.Image object>
[Finished in 8.3s]
'''
#?????????

















'''

# Vector Representations of Words


import numpy
import math
import tensorflow as tf



# 嵌套参数矩阵。用唯一的随机值来初始化这个大矩阵。
embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
# 输出权重, 与之对应的 输入嵌套值
nce_weights = tf.Variable(tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0/math.sqrt(embedding_size)))
nce_biases = tf.Variable(tf.zeros([vocabulary_size]))
# 建立输入占位符
train_inputs = tf.placeholder(tf.int32, shape=[batch_size])
train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
# 对批数据中的单词建立嵌套向量
embed = tf.nn.embedding_lookup(embeddings, train_inputs)
# 使用噪声-比对的训练方式来预测目标单词
# 计算 NCE 损失函数, 每次使用负标签的样本.
loss = tf.reduce_mean(tf.nn.nce_loss(nce_weights, nce_biases, embed, train_labels, num_sampled, vocabulary_size))
# 使用 SGD 控制器.
optimizer = tf.train.GradientDescentOptimizer(learning_rate=1.0).minimize(loss)

# train model
for inputs,labels in generate_batch():
	feed_dict = {training_inputs:inputs, training_labels:labels}
	_, cur_loss = session.run([optimizer, loss], feed_dict=feed_dict)


'''





