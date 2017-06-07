# coding: utf-8
# 线性混合模型    固定和随机效应的线性建模。

# https://edward-cn.readthedocs.io/zh/latest/Tutorials/tutorials/
# http://edwardlib.org/tutorials/linear-mixed-effects-models



from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import edward as ed 
import pandas as pd 
import tensorflow as tf 
import matplotlib.pyplot as plt 

from edward.models import Normal

plt.style.use('ggplot')
ed.set_seed(42)


# s - students -1:2972
# d - instructors - codes that need to be remapped
# dept also needs to be remapped
data = pd.read_csv('../examples/data/insteval.csv')
data['dcodes'] = data['d'].astype('category').cat.codes
data['deptcodes'] = data['dept'].astype('category').cat.codes
data['s'] = data['s'] - 1

train = data.sample(frac=0.8)
test = data.drop(train.index)

train.head()




