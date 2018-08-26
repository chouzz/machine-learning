# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 11:09:43 2018

@author: hua
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers import Dense,Flatten,Conv2D,MaxPool2D,Activation
import numpy as np

# read data
trainpath = r'E:\nfsroot\project\kaggle\Digit Recognizer\data\train.csv'
testpath = r'E:\nfsroot\project\kaggle\Digit Recognizer\data\test.csv'
data = pd.read_csv(trainpath)

# data precessing
train = data.iloc[:10000,1:].values/255
test = data.iloc[:10000,:1].values
train = train.reshape(-1,1,28,28)
test = np_utils.to_categorical(test,num_classes=10)
X_train,X_test,y_train,y_test = train_test_split(train,test,test_size=0.25,random_state=123)

import gc
del data,train,test
gc.collect()

# create model

# Conv layer
model = Sequential()
model.add(Conv2D(
        batch_input_shape=(None,1,28,28),
        filters=32,
        kernel_size=5,
        strides=1,
        padding='same',
        data_format='channels_first',
        ))
model.add(Activation('relu'))
# Pooling layer
model.add(MaxPool2D(
        pool_size=2,
        strides=2,
        padding='same',
        data_format='channels_first',
        ))

# Conv layer 2
model.add(Conv2D(
        64,
        5,
        strides=1,
        padding='same',
        data_format='channels_first',
        ))
model.add(Activation('relu'))

# pooling layer 2
model.add(MaxPool2D(
        2,
        2,
        padding='same',
        data_format='channels_first',
        ))

# fully connected layer
model.add(Flatten())
model.add(Dense(1024))
model.add(Activation('relu'))

# fully connneted layer to 10 classes
model.add(Dense(10))
model.add(Activation('softmax'))

# Define optimizer
adam = Adam(lr=0.001)

# Compile Model
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

# train model
print('training-----')
model.fit(X_train,y_train,epochs=1,steps_per_epoch=64)

# test model
model.predict(X_test)
loss,accuracy = model.evaluate(X_test,y_test)

print('\nloss',loss)
print('\naccuracy',accuracy)






