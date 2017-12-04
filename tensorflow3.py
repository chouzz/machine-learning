# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:42:31 2017

@author: hua
"""

import tensorflow as tf

state = tf.Variable(5,name='counter')
#print(state.name)
one = tf.constant(1) 

new_value = tf.add(state,one)  #state+one 变量加上常量等于变量
update = tf.assign(state,new_value) #把new_value加载到state，此刻他们相等

init = tf.initialize_all_variables()#最重要，初始化所有变量，还没有激活，要用session.run激活变量

with tf.Session() as sess:
    sess.run(init)
    for _ in range(3):
        sess.run(update)
        print(sess.run(state))