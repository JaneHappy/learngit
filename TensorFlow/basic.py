# coding: utf-8

import tensorflow as tf 




# part 0: install

hello = tf.constant('Hello, TensorFlow!')
with tf.Session() as sess:
	print sess.run(hello)
	a = tf.constant(10)
	b = tf.constant(32)
	print a,'+',b,'=',sess.run(a+b)
# Hello, TensorFlow!
# Tensor("Const_1:0", shape=(), dtype=int32) + Tensor("Const_2:0", shape=(), dtype=int32) = 42






# part 1

matrix1 = tf.constant([[3., 3.]])
matrix2 = tf.constant([[2.],[2.]])
product = tf.matmul(matrix1, matrix2)
'''
sess = tf.Session()
result = sess.run(product)
print result
sess.close()
'''
with tf.Session() as sess:
	result = sess.run(product) #([product])
	print result



# part 2

sess = tf.InteractiveSession()
x = tf.Variable([1.0, 2.0])
a = tf.constant([3.0, 3.0])
x.initializer.run()
sub = tf.subtract(x,a)
print sub.eval()
sess.close()



# part 3

state = tf.Variable(0, name="counter")
one = tf.constant(1)
new_value = tf.add(state, one)
update = tf.assign(state, new_value)
init_op = tf.initialize_all_variables()
with tf.Session() as sess:
	sess.run(init_op)
	print "init",'  state', sess.run(state)
	for i in range(3): #for _ in range(3):
		sess.run(update)
		print "Time",i, 'state', sess.run(state)



# part 4: Fetch

input1 = tf.constant(3.0)
input2 = tf.constant(2.0)
input3 = tf.constant(5.0)
intermed = tf.add(input2, input3)
mul = tf.multiply(input1, intermed)
with tf.Session() as sess:
	result = sess.run([mul, intermed])
	print result
# [21.0, 7.0]



# part 5: Feed

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.multiply(input1, input2)
with tf.Session() as sess:
	print sess.run([output], feed_dict={input1:[7.], input2:[2.]})
# [array([ 14.], dtype=float32)]




