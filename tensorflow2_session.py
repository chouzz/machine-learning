# -*- coding: utf-8 -*-
"""
Created on Mon Nov 27 22:04:47 2017

@author: hua
"""
#会话控制

import tensorflow as tf


matrix1 = tf.constant([[3,3]])
matrix2 = tf.constant([[2],[2]])  #1行2列矩阵和2行1列矩阵相乘，结果为12.

product = tf.matmul(matrix1,matrix2)  #matrix multiply 矩阵乘法和np.dot(m1,m2)相似

##method 1
#sess = tf.Session()   #Session是一个对象，首字母要大写
#result = sess.run(product)  #结果为12
#print(result)
#sess.close()

#method 2
with tf.Session() as sess:  #不用关闭session
    result2 = sess.run(product)
    print(result2)
    