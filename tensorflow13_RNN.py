import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#this is data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

#hyperparameters
lr = 0.001
training_iters = 100000
batch_size = 128

n_inputs = 28 #MNIST data input (img shape:28*28)
n_steps = 28 #time steps
n_hidden_dnis = 128     #neurons in hidden layer
n_classes = 10  #MNIST classes(0-9 digits)


#tf Graph input
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

#define weights
