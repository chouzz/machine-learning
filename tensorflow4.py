# -*- coding: utf-8 -*-
"""
Created on Fri Dec  1 22:23:39 2017

@author: hua
"""

import tensorflow as tf

input1 = tf.placeholder(tf.float32)  #float32为类型
input2 = tf.placeholder(tf.float32)

output = tf.mul(input1,intput2)  #乘法运算

with tf.Session() as sess:
    print(sess.run(output,))