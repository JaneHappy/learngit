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
qW_1 = Normal(tf.Variable(tf.zeros([2,1])), tf.nn.softplus(tf.Variable(tf.zeros([2,1]))))
qb_0 = Normal(tf.Variable(tf.zeros(2)), tf.nn.softplus(tf.Variable(tf.zeros(2))))
qb_1 = Normal(tf.Variable(tf.zeros(1)), tf.nn.softplus(tf.Variable(tf.zeros(1))))


import edward as ed 



'''
qW_1 = Normal(tf.Variable(tf.zeros([1,2])), tf.nn.softplus(tf.Variable(tf.zeros([2,1]))))
print W_0.shape, W_1.shape, b_0.shape, b_1.shape
print qW_0.shape, qW_1.shape, qb_0.shape, qb_1.shape
print x.shape,y.shape
(1, 2) (2, 1) (2,) (1,)
(1, 2) (2, 2) (2,) (1,)
(50, 1) (50, 1)
[Finished in 6.5s]
'''


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



inference = ed.KLqp({W_0: qW_0, b_0:qb_0, W_1: qW_1, b_1: qb_1}, data={y: y_trn}) #{y: y_trn}) #
inference.run(n_iter=500)

'''
2017-06-07 19:44:30.795126: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.1 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-07 19:44:30.795323: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use SSE4.2 instructions, but these are available on your machine and could speed up CPU computations.
2017-06-07 19:44:30.795333: W tensorflow/core/platform/cpu_feature_guard.cc:45] The TensorFlow library wasn't compiled to use AVX instructions, but these are available on your machine and could speed up CPU computations.

  1/500 [  0%]                                ETA: 829s | Loss: 1789.927
  5/500 [  1%]                                ETA: 166s | Loss: 1571.937
 10/500 [  2%]                                ETA: 83s | Loss: 5036.991 
 15/500 [  3%]                                ETA: 55s | Loss: 2620.026
 20/500 [  4%] █                              ETA: 41s | Loss: 1308.074
 25/500 [  5%] █                              ETA: 33s | Loss: 1204.835
 30/500 [  6%] █                              ETA: 27s | Loss: 997.456 
 40/500 [  8%] ██                             ETA: 20s | Loss: 1217.202
 45/500 [  9%] ██                             ETA: 18s | Loss: 1117.022
 50/500 [ 10%] ███                            ETA: 16s | Loss: 1120.087
 55/500 [ 11%] ███                            ETA: 14s | Loss: 1052.249
 60/500 [ 12%] ███                            ETA: 13s | Loss: 998.973 
 65/500 [ 13%] ███                            ETA: 12s | Loss: 1078.277
 75/500 [ 15%] ████                           ETA: 10s | Loss: 821.691
 80/500 [ 16%] ████                           ETA: 10s | Loss: 997.204
 85/500 [ 17%] █████                          ETA: 9s | Loss: 864.562
 90/500 [ 18%] █████                          ETA: 8s | Loss: 955.535
 95/500 [ 19%] █████                          ETA: 8s | Loss: 608.964
100/500 [ 20%] ██████                         ETA: 8s | Loss: 779.940
105/500 [ 21%] ██████                         ETA: 7s | Loss: 483.508
110/500 [ 22%] ██████                         ETA: 7s | Loss: 658.378
115/500 [ 23%] ██████                         ETA: 6s | Loss: 480.328
120/500 [ 24%] ███████                        ETA: 6s | Loss: 250.548
125/500 [ 25%] ███████                        ETA: 6s | Loss: 317.656
130/500 [ 26%] ███████                        ETA: 6s | Loss: 341.068
135/500 [ 27%] ████████                       ETA: 5s | Loss: 123.996
140/500 [ 28%] ████████                       ETA: 5s | Loss: 157.485
145/500 [ 28%] ████████                       ETA: 5s | Loss: 177.355
150/500 [ 30%] █████████                      ETA: 5s | Loss: 61.861
155/500 [ 31%] █████████                      ETA: 4s | Loss: 14.024
160/500 [ 32%] █████████                      ETA: 4s | Loss: 2.071 
165/500 [ 33%] █████████                      ETA: 4s | Loss: 117.157
170/500 [ 34%] ██████████                     ETA: 4s | Loss: -3.162
175/500 [ 35%] ██████████                     ETA: 4s | Loss: 50.118
180/500 [ 36%] ██████████                     ETA: 4s | Loss: 57.768
185/500 [ 37%] ███████████                    ETA: 3s | Loss: 57.777
190/500 [ 38%] ███████████                    ETA: 3s | Loss: 52.555
195/500 [ 39%] ███████████                    ETA: 3s | Loss: 26.809
200/500 [ 40%] ████████████                   ETA: 3s | Loss: 16.164
205/500 [ 41%] ████████████                   ETA: 3s | Loss: 65.102
210/500 [ 42%] ████████████                   ETA: 3s | Loss: 5.457 
215/500 [ 43%] ████████████                   ETA: 3s | Loss: -5.716
220/500 [ 44%] █████████████                  ETA: 3s | Loss: 13.438
225/500 [ 45%] █████████████                  ETA: 3s | Loss: -1.699
230/500 [ 46%] █████████████                  ETA: 2s | Loss: 8.506 
235/500 [ 47%] ██████████████                 ETA: 2s | Loss: -4.682
240/500 [ 48%] ██████████████                 ETA: 2s | Loss: 9.381 
245/500 [ 49%] ██████████████                 ETA: 2s | Loss: -6.466
250/500 [ 50%] ███████████████                ETA: 2s | Loss: 37.004
255/500 [ 51%] ███████████████                ETA: 2s | Loss: 61.472
260/500 [ 52%] ███████████████                ETA: 2s | Loss: 66.571
265/500 [ 53%] ███████████████                ETA: 2s | Loss: 16.035
270/500 [ 54%] ████████████████               ETA: 2s | Loss: -4.776
275/500 [ 55%] ████████████████               ETA: 2s | Loss: 12.611
280/500 [ 56%] ████████████████               ETA: 2s | Loss: -5.850
285/500 [ 56%] █████████████████              ETA: 2s | Loss: -5.080
290/500 [ 57%] █████████████████              ETA: 1s | Loss: 7.643 
295/500 [ 59%] █████████████████              ETA: 1s | Loss: 57.405
300/500 [ 60%] ██████████████████             ETA: 1s | Loss: 3.328
305/500 [ 61%] ██████████████████             ETA: 1s | Loss: 25.950
310/500 [ 62%] ██████████████████             ETA: 1s | Loss: 20.137
315/500 [ 63%] ██████████████████             ETA: 1s | Loss: 16.025
320/500 [ 64%] ███████████████████            ETA: 1s | Loss: 19.342
325/500 [ 65%] ███████████████████            ETA: 1s | Loss: 0.691 
335/500 [ 67%] ████████████████████           ETA: 1s | Loss: 32.863
345/500 [ 69%] ████████████████████           ETA: 1s | Loss: 19.391
350/500 [ 70%] █████████████████████          ETA: 1s | Loss: 16.065
355/500 [ 71%] █████████████████████          ETA: 1s | Loss: -11.864
360/500 [ 72%] █████████████████████          ETA: 1s | Loss: 30.690 
365/500 [ 73%] █████████████████████          ETA: 1s | Loss: -5.281
370/500 [ 74%] ██████████████████████         ETA: 1s | Loss: 14.484
375/500 [ 75%] ██████████████████████         ETA: 1s | Loss: -1.745
380/500 [ 76%] ██████████████████████         ETA: 0s | Loss: -9.917
385/500 [ 77%] ███████████████████████        ETA: 0s | Loss: -7.986
390/500 [ 78%] ███████████████████████        ETA: 0s | Loss: -4.829
395/500 [ 79%] ███████████████████████        ETA: 0s | Loss: -3.970
400/500 [ 80%] ████████████████████████       ETA: 0s | Loss: 12.318
405/500 [ 81%] ████████████████████████       ETA: 0s | Loss: -6.047
410/500 [ 82%] ████████████████████████       ETA: 0s | Loss: 17.054
415/500 [ 83%] ████████████████████████       ETA: 0s | Loss: -1.409
420/500 [ 84%] █████████████████████████      ETA: 0s | Loss: 5.780
425/500 [ 85%] █████████████████████████      ETA: 0s | Loss: -3.180
430/500 [ 86%] █████████████████████████      ETA: 0s | Loss: 32.764
435/500 [ 87%] ██████████████████████████     ETA: 0s | Loss: -6.682
440/500 [ 88%] ██████████████████████████     ETA: 0s | Loss: 17.119
445/500 [ 89%] ██████████████████████████     ETA: 0s | Loss: 23.543
450/500 [ 90%] ███████████████████████████    ETA: 0s | Loss: -5.742
455/500 [ 91%] ███████████████████████████    ETA: 0s | Loss: -3.671
460/500 [ 92%] ███████████████████████████    ETA: 0s | Loss: 11.476
465/500 [ 93%] ███████████████████████████    ETA: 0s | Loss: 10.519
470/500 [ 94%] ████████████████████████████   ETA: 0s | Loss: 34.492
475/500 [ 95%] ████████████████████████████   ETA: 0s | Loss: 24.629
480/500 [ 96%] ████████████████████████████   ETA: 0s | Loss: 8.961 
485/500 [ 97%] █████████████████████████████  ETA: 0s | Loss: -4.909
490/500 [ 98%] █████████████████████████████  ETA: 0s | Loss: 24.882
495/500 [ 99%] █████████████████████████████  ETA: 0s | Loss: -6.525
500/500 [100%] ██████████████████████████████ Elapsed: 3s | Loss: -7.001
[Finished in 20.6s]
'''




