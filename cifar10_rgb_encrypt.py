#!/usr/bin/env python

from __future__ import print_function
import time
import numpy as np
import sys
import theano
import theano.tensor as T
import lasagne
from PIL import Image
#from Crypto.Cipher import AES
import hashlib
import binascii
from sklearn import preprocessing
from sklearn.decomposition import PCA
# In[11]:
sys.path.append('/u/mudumbas/cifar10_vgg')
from paillier.paillier import *

from cifar10_data import load_cifar10
import lasagne_trainer

import matplotlib.pyplot as plt
priv, pub = generate_keypair(128)
X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10()
print('Train data shape: ', X_train.shape)
print('Train labels shape: ', y_train.shape)
print('Validation data shape: ', X_val.shape)
print('Validation labels shape: ', y_val.shape)
print('Test data shape: ', X_test.shape)
print('Test labels shape: ', y_test.shape)
X_train=X_train.reshape(X_train.shape[0],X_train.shape[1],-1)
X_test=X_test.reshape(X_test.shape[0],X_test.shape[1],-1)
X_val=X_val.reshape(X_val.shape[0],X_val.shape[1],-1)

X_train=encrypt(pub,X_train)
X_test=encrypt(pub,X_test)
X_val=encrypt(pub,X_val)

for i in range(X_train.shape[0]):
    for j in range(X_train.shape[1]):
        for k in range(X_train.shape[2]):
            X_train[i,j,k]=np.log(X_train[i,j,k])
for i in range(X_test.shape[0]):
    for j in range(X_test.shape[1]):
        for k in range(X_test.shape[2]):
            X_test[i,j,k]=np.log(X_test[i,j,k])

for i in range(X_val.shape[0]):
    for j in range(X_val.shape[1]):
        for k in range(X_val.shape[2]):
            X_val[i,j,k]=np.log(X_val[i,j,k])

X_train=(X_train-np.mean(X_train,axis=0))/np.std(X_train)
X_test=(X_test-np.mean(X_test,axis=0))/np.std(X_test)
X_val=(X_val-np.mean(X_val,axis=0))/np.std(X_val)

tol = 1e-12
X_train[abs(X_train)<tol]=0.0
X_test[abs(X_test)<tol]=0.0
X_val[abs(X_val)<tol]=0.0


X_train=X_train.astype(np.float32)
X_test=X_test.astype(np.float32)
X_val=X_val.astype(np.float32)

X_train=X_train.reshape(-1,3,32,32)
X_test=X_test.reshape(-1,3,32,32)
X_val=X_val.reshape(-1,3,32,32)

input_var = T.tensor4('inputs')

def create_v3(input_var, input_shape=(3, 32, 32),
              ccp_num_filters=[64, 128], ccp_filter_size=3,
              fc_num_units=[128, 128], num_classes=10,
              **junk):
    # input layer
    network = lasagne.layers.InputLayer(shape=(None,) + input_shape,
                                        input_var=input_var)
    # conv-relu-conv-relu-pool layers
    for num_filters in ccp_num_filters:
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=num_filters,
            filter_size=(ccp_filter_size, ccp_filter_size),
            pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=num_filters,
            filter_size=(ccp_filter_size, ccp_filter_size),
            pad='same',
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
        network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
    # fc-relu
    for num_units in fc_num_units:
        network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=num_units,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform(gain='relu'))
    # output layer
    network = lasagne.layers.DenseLayer(
        lasagne.layers.dropout(network, p=.5),
        num_units=num_classes,
        nonlinearity=lasagne.nonlinearities.softmax)
    return network

param = dict(ccp_num_filters=[64, 128], ccp_filter_size=3,
             fc_num_units=[256, 256], num_classes=10,
             learning_rate=1e-2, learning_rate_decay=0.5,
             momentum=0.9, momentum_decay=0.5,
             decay_after_epochs=10,
             batch_size=128, num_epochs=50)

network = create_v3(input_var, **param)
model, loss_history, train_acc_history, val_acc_history = lasagne_trainer.train(
    network, input_var, X_train, y_train, X_val, y_val,
    learning_rate=param['learning_rate'], learning_rate_decay=param['learning_rate_decay'],
    momentum=param['momentum'], momentum_decay=param['momentum_decay'],
    decay_after_epochs=param['decay_after_epochs'],
    batch_size=param['batch_size'], num_epochs=param['num_epochs'],
    save_path='net_v3')
print('%.3f' % min(loss_history), max(train_acc_history), max(val_acc_history), \
    ' '.join('%s=%s' % (k,param[k]) for k in param))
