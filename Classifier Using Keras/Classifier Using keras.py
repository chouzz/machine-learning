# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 15:52:03 2018

@author: hua
"""
# using MINIST data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense,Activation
from keras.utils import np_utils
from keras.optimizers import RMSprop
from sklearn.model_selection import train_test_split
import gc

# read data
trainpath = r'E:\nfsroot\project\kaggle\Digit Recognizer\data\train.csv'
testpath = r'E:\nfsroot\project\kaggle\Digit Recognizer\data\test.csv'
data = pd.read_csv(trainpath)

# data processing
train = data.iloc[:,1:]/255.
label = data.iloc[:,:1]

X_train,X_test,y_train,y_test = train_test_split(train,label,test_size=0.2,\
                                        random_state=10)
                                            
y_train = np_utils.to_categorical(y_train,num_classes=10)
y_test = np_utils.to_categorical(y_test,num_classes=10)
img = train.iloc[0,:].values.reshape([28,28])
plt.imshow(img,cmap='binary')
plt.title(label.iloc[0])
plt.show()

del data,train,label
gc.collect()  
# create model
model = Sequential()
model.add(Dense(input_dim=784,output_dim=32))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

# optimizers 
rmsprop = RMSprop(lr=0.001)

# compolie model
model.compile(optimizer=rmsprop,loss='categorical_crossentropy',metrics=['accuracy'])
# training

model.fit(X_train, y_train, epochs=2, steps_per_epoch=32)

# test model
print('testing-----')
loss,accuracy = model.evaluate(X_test,y_test)

print('test loss',loss)
print('test accuracy',accuracy)



