# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 21:36:17 2017

@author: hua
"""

import tensorflow as tf
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)#生成float32类型的随机数，目的是创造数据
y_data = x_data*0.1 + 0.3

#create tensorflow structure start
Weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))#一维矩阵，随机数列生成，生成1维结构，范围-1到1
biases = tf.Variable(tf.zeros([1]))#初始值给0



y = Weights*x_data + biases #预测值y 

loss = tf.reduce_mean(tf.square(y-y_data))#预测值和真实值之间的差值
optimizer = tf.train.GradientDescentOptimizer(0.5)#建立优化器，目的是减小误差，提升参数准确度
train = optimizer.minimize(loss)

init = tf.initialize_all_variables()#初始化变量

sess = tf.Session()
sess.run(init)      #Very important 激活init，激活整个神经网络


for step in range(201):
    sess.run(train)
    if step % 10 == 0: #每隔20步打印结果
        print(step,sess.run(Weights),sess.run(biases))

