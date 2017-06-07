# coding: utf-8
# 线性混合模型    固定和随机效应的线性建模。

# https://edward-cn.readthedocs.io/zh/latest/Tutorials/tutorials/
# http://edwardlib.org/tutorials/linear-mixed-effects-models

# http://nbviewer.jupyter.org/github/blei-lab/edward/blob/master/notebooks/linear_mixed_effects_models.ipynb



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



s_train = train['s'].values.astype(int)
d_train = train['dcodes'].values.astype(int)
dept_train = train['deptcodes'].values.astype(int)
y_train = train['y'].values.astype(float)
service_train = train['service'].values.astype(int)
n_obs_train = train.shape[0]

s_test = test['s'].values.astype(int)
d_test = test['dcodes'].values.astype(int)
dept_test = test['deptcodes'].values.astype(int)
y_test = test['y'].values.astype(float)
service_test = test['service'].values.astype(int)
n_obs_test = test.shape[0]

n_s = 2972  # number of students
n_d = 1128  # number of instructors
n_dept = 14  # number of departments
n_obs = train.shape[0]  # number of observations



# Model


