# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 13:49:41 2017

@author: hua
"""

import tensorflow as tf

def add_layer(inputs,in_size,out_size,activate_function=None):#定义一个神经层的函数
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))#生成随机矩阵
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)#类似列表的东西
    Wx_plus_b = tf.matmul(inputs,Weights)+biases
    if avtivation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activate_function(Wx_plus_b)
    return outputs#矩阵乘法，还没激活的值存在这里