

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#number 1 to 10 data
mnist = input_data.read_data_sets('MNIST_data',one_hot=True)#如果没有数据包就从网上下载下来

def add_layer(inputs,in_size,out_size,activation_function=None):#定义一个神经层的函数
    Weights = tf.Variable(tf.random_normal([in_size,out_size]))#生成随机矩阵
    biases = tf.Variable(tf.zeros([1,out_size])+0.1)#类似列表的东西
    Wx_plus_b = tf.matmul(inputs,Weights) + biases #矩阵乘法，还没激活的值存在这里
    if activation_function is None:
        outputs = Wx_plus_b#线性函数，直接输出
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs, v_ys):
    global prediction #定义全局变量
    y_pre = sess.run(prediction, feed_dict={xs:v_xs})
    correct_prediction = tf.equal(tf.argmax(y_pre, 1),tf.argmax(v_ys, 1))#预测数据和真实数据想比较
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))#预测精度
    result = sess.run(accuracy, feed_dict={xs:v_xs, ys:v_ys})
    return result

#define placeholder for inputs to network
xs = tf.placeholder(tf.float32,[None, 784])#28*28
ys = tf.placeholder(tf.float32,[None, 10])

#add output layer
prediction = add_layer(xs,784,10, activation_function=tf.nn.softmax)#softmax is activation function

#the error between prediction and real data
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(prediction),
                                              reduction_indices=[1]))  #loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
#import step
init = tf.global_variables_initializer()
sess.run(init)

for i in range(1000):
    batch_xs,batch_ys = mnist.train.next_batch(100)#提取100个数据来学习
    sess.run(train_step, feed_dict={xs: batch_xs, ys: batch_ys})
    if i%50 == 0:
        print(compute_accuracy(
            mnist.test.images, mnist.test.labels))
        
