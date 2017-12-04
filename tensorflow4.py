# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 22:23:39 2017

@author: hua
"""

import tensorflow as tf

input1 = tf.placeholder(tf.float32)  #大部分时候是float32为类型
input2 = tf.placeholder(tf.float32)

output = tf.multiply(input1,input2) #乘法运算，新版本tensorflow不再使用.mul命令,而是用.multiply

with tf.Session() as sess:
    #print(sess.run(output,))是一般结构，因为有placeholder所以要以dictionary传入值
    print(sess.run(output,feed_dict={input1:[7.],input2:[2.]}))
    #用了placeholder就必须用feed_dict