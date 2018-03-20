# -*- coding: utf-8 -*-
"""
Created on Tue Dec  5 14:20:30 2017

@author: hua
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs,in_size,out_size,activation_function=None):#定义一个神经层的函数
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))#生成随机矩阵
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)#类似列表的东西
    Wx_plus_b = tf.matmul(inputs,Weights) + biases #矩阵乘法，还没激活的值存在这里
    if activation_function is None:
        outputs = Wx_plus_b#线性函数，直接输出
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data = np.linspace(-1,1,300)[:,np.newaxis]#-1到1之间，维度为300
noise = np.random.normal(0,0.05,x_data.shape)#方差0.05 ，格式是x_data形式
y_data = np.square(x_data)-0.5 + noise#加上噪点，不完全是二次函数

xs = tf.placeholder(tf.float32,[None,1]) 
ys = tf.placeholder(tf.float32,[None,1])
#xs ys 要给train_step的值，None是指无论给多少个例子都可以
l1 = add_layer(xs,1,10,activation_function=tf.nn.relu)
#输入为x_data，输入层为1，中间层为10，使用通常的激活函数nn.relu激活
prediction = add_layer(l1,10,1,activation_function=None)
#输入为中间层的输出，输出为1个神经元，激活函数为线性函数

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data - prediction),
                                    reduction_indices=[1]))
#square平方，reduce_sum求和，reduce_mean求平均
train_step = tf.train.GradientDescentOptimizer(0.3).minimize(loss)
#训练步骤，使用最常用的优化器，0.1为学习效率，优化器是要使loss最小

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

fig = plt.figure()#生成一个图片框
ax = fig.add_subplot(1,1,1)#连续性画图，编号1,1,1,
ax.scatter(x_data,y_data)#以点的形式再图中显示
plt.ion()#在show（）之后，程序会暂停，这句是让程序不暂停
plt.show()#打印输出￼

for i in range(1000):#重复1000次
    sess.run(train_step,feed_dict={xs:x_data,ys:y_data})
    if i%50 == 0:
        #print(sess.run(loss,feed_dict={xs:x_data,ys:y_data}))
        try:
            ax.lines.remove(lines[0])#擦除第一条线
        except Exception:
            pass
        prediction_value = sess.run(prediction,feed_dict={xs:x_data})
        lines = ax.plot(x_data,prediction_value,'r-',lw=5)#将prediction_value plot上去,曲线形式，x轴x_data，y轴value，
        
        #红色为线，宽度为5
        plt.pause(0.4)#暂停0.1秒
        
        
             
