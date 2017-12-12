import tensorflow as tf
from sklearn.datases import load_digits
from sklearn.cross_calidation import train_test_split
from sklearn.preprocessing import LabelBinarizer

#load data
digits = load_digits()
X = digits.data  #0 to 9 picture data
y = digits.target #become binary data, for example,if the data is 2, y is '01000'
y = LabelBinarizer().fit_transform(y)
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size = .3)
#split train data and test data\


def add_layer(inputs, in_size, out_size, layer_name, activation_function = None):
    #add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs,Weights) + biases

    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b,)
        tf.histogram_summary(layer_name + '/outputs', outputs)
