# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 18:33:34 2017

@author: hua
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:20:30 2017

@author: hua
"""

import tensorflow as tf
import numpy as np

def add_layer(inputs,in_size,out_size,activation_function=None):#定义一个神经层的函数
    with tf.name_scope('layer'):  
        with tf.name_scope('Weights'):  
            Weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')#生成随机矩阵
        with tf.name_scope('biases'): 
            biases = tf.Variable(tf.zeros([1,out_size])+0.1)#类似列表的东西
        with tf.name_scope('Wx_plus_b'): 
            Wx_plus_b = tf.matmul(inputs,Weights) + biases #矩阵乘法，还没激活的值存在这里
        if activation_function is None:
            outputs = Wx_plus_b#线性函数，直接输出
        else:
            outputs = activation_function(Wx_plus_b)
        return outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis]#-1到1之间，维度为300
noise = np.random.normal(0,0.05,x_data.shape)#方差0.05 ，格式是x_data形式
y_data = np.square(x_data)-0.5 + noise#加上噪点，不完全是二次函数

with tf.name_scope('inputs'):  
    xs = tf.placeholder(tf.float32,[None,1],name='x_input')
    ys = tf.placeholder(tf.float32,[None,1],name='y_input')
#xs ys 要给train_step的值，None是指无论给多少个例子都可以
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
#输入为x_data，输入层为1，中间层为10，使用通常的激活函数nn.relu激活
prediction = add_layer(l1,10,1,activation_function=None)
#输入为中间层的输出，输出为1个神经元，激活函数为线性函数

with tf.name_scope('loss'): 
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data - prediction),
                                    reduction_indices=[1]))
#square平方，reduce_sum求和，reduce_mean求平均
with tf.name_scope('train'): 
    train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)
#训练步骤，使用最常用的优化器，0.1为学习效率，优化器是要使loss最小

init = tf.initialize_all_variables()
sess = tf.Session()
writer = tf.summary.FileWriter("d://logs",sess.graph)#有文件，但无法显示
sess.run(init)

