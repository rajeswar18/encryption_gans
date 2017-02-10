#!/usr/bin/env python
# coding: utf-8
import pickle
import gzip
import numpy as np
import lasagne
import PIL.Image as Image
import theano.tensor as T
import theano
import os
import sys
import PIL
import math
from PIL import Image
from Crypto.Cipher import AES
import hashlib
import binascii
from sklearn import preprocessing
from sklearn.decomposition import PCA
# In[11]:
sys.path.append('/u/mudumbas/encryption')
from paillier.paillier import *


data = pickle.load(gzip.open('mnist.pkl.gz', 'r'))
train, valid, test=data
xtrain, ytrain = train
xtest, ytest=test
xtrain=xtrain.reshape(-1,1,28,28)
xtest=xtest.reshape(-1,1,28,28)
ytrain=ytrain.astype(np.int32)
ytest=ytest.astype(np.int32)
batch_size=64
num_epochs=50
xtrain=xtrain*255
xtrain=xtrain.astype(np.int32)
xtest=xtest*255
xtest=xtest.astype(np.int32)

#priv, pub = generate_keypair(256)
global password
password = hashlib.sha256("dummy is the".encode('utf-8')).digest()
def encrypt_aes(xtrain):
    xtrain_encrypt=np.zeros_like(xtrain)
    for i1 in range(xtrain.shape[0]):

        # initialize variables
        plaintext = list()
        plaintextstr = ""
        im=xtrain[i1].reshape(xtrain.shape[2],xtrain.shape[3])
        #print im.shape
        width = im.shape[0]
        height = im.shape[1]
        # break up the image into a list, each with pixel values and then append to a string
        for y in range(0,height):
            #print("Row: %d") %y  # print row number
            for x in range(0,width):
                #print pix[x,y]  # print each pixel RGB tuple
                plaintext.append(im[x,y])

        # add 100 to each tuple value to make sure each are 3 digits long.  being able to do this is really just a PoC
        # that you'll be able to use a raw application of RSA to encrypt, rather than PyCrypto if you wanted.
        for i in range(0,len(plaintext)):
                plaintextstr = plaintextstr + "%d" %(int(plaintext[i])+100)


        # length save for encrypted image reconstruction
        relength = len(plaintext)

        # append dimensions of image for reconstruction after decryption
        plaintextstr += "h" + str(height) + "h" + "w" + str(width) + "w"

        # make sure that plantextstr length is a multiple of 16 for AES.  if not, append "n".  not safe in theory
        # and i should probably replace this with an initialization vector IV = 16 * '\x00' at some point.  In practice
        # this IV buffer should be random.
        while (len(plaintextstr) % 16 != 0):
            plaintextstr = plaintextstr + "n"

        # encrypt plaintext
        obj = AES.new(password, AES.MODE_CBC, 'This is an IV456')
        ciphertext = obj.encrypt(plaintextstr)
        #ciphertext = binascii.hexlify(ciphertext)

        #imagedata=[int.from_bytes(ciphertext[i:i+3],byteorder='big') for i in range(0, 3*int(relength), 3)]
        imagedata=[int(ciphertext[i:i+3].encode('hex'), 16) for i in range(0, 3*int(relength), 3)]

        ct=0
        newim=np.zeros([width,height],dtype='object')
        for y in range(0,height):
            #print("Row: %d") %y  # print row number
            for x in range(0,width):
                #print pix[x,y]  # print each pixel RGB tuple
                newim[x,y] = imagedata[ct]
                ct=ct+1
        #newim = preprocessing.scale(newim)

        xtrain_encrypt[i1]=newim.reshape(1,height,width)
    return xtrain_encrypt

xtrain=encrypt_aes(xtrain)
xtest=encrypt_aes(xtest)
xtrain=xtrain.astype(np.float32)
xtest=xtest.astype(np.float32)
xtrain=xtrain.reshape(-1,784)
xtest=xtest.reshape(-1,784)





print "before scaling"

print xtrain[1]


print xtrain[1]
xtrain=(xtrain-np.mean(xtrain,axis=0))/np.std(xtrain)
xtest=(xtest-np.mean(xtest,axis=0))/np.std(xtrain)
tol = 1e-8
xtrain[abs(xtrain)<tol]=0.0
xtest[abs(xtest)<tol]=0.0

xtrain=xtrain.astype(np.float32)
xtest=xtest.astype(np.float32)
print "after enc scaling"

print xtrain[1]
#xtrain=xtrain.reshape(-1,784)
#xtest=xtest.reshape(-1,784)



xtrain=xtrain.reshape(-1,1,28,28)
xtest=xtest.reshape(-1,1,28,28)
print xtrain.shape
print "done"

dsize=(None,1,28,28)
out_size=10
x=T.tensor4('inputs')
y=T.ivector('labels')


    # In[19]:
def make_network(input_var):
    network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
                                        input_var=input_var)
    # This time we do not apply input dropout, as it tends to work less well
    # for convolutional layers.

    # Convolutional layer with 32 kernels of size 5x5. Strided and padded
    # convolutions are supported as well; see the docstring.
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    # Expert note: Lasagne provides alternative convolutional layers that
    # override Theano's choice of which implementation to use; for details
    # please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

    # Max-pooling layer of factor 2 in both dimensions:
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # Another convolution with 32 5x5 kernels, and another 2x2 pooling:
    network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(5, 5),
            nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

    # A fully-connected layer of 256 units with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=256,
            nonlinearity=lasagne.nonlinearities.rectify)

    # And, finally, the 10-unit output layer with 50% dropout on its inputs:
    network = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(network, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return network


# In[23]:

network = make_network(x)
#learning_rate = T.scalar(name='learning_rate')
weight_decay = 1e-5
predictions=lasagne.layers.get_output(network)
cost=lasagne.objectives.categorical_crossentropy(predictions,y)
cost=cost.mean()
params=lasagne.layers.get_all_params(network)
#updates=lasagne.updates.sgd(cost,params,learning_rate=lr)
updates = lasagne.updates.nesterov_momentum(cost, params, learning_rate=0.01, momentum=0.8)
base_lr = 1e-2
lr_decay = 0.01


# In[25]:

train_fun = theano.function([x,y],cost,updates=updates)


# In[26]:

predict=lasagne.layers.get_output(network,deterministic=True)
loss=lasagne.objectives.categorical_crossentropy(predictions,y)
loss=loss.mean()
acc=T.mean(T.eq(T.argmax(predict,axis=1),y),dtype=theano.config.floatX)
acc_fun=theano.function([x,y],acc)
num_batches=xtrain.shape[0]/batch_size
for i in range(num_epochs):
    #lr = base_lr * (lr_decay ** i)
    for j in range(num_batches):
        xtrain_batch=xtrain[j*batch_size:(j+1)*batch_size]
        ytrain_batch=ytrain[j*batch_size:(j+1)*batch_size]
        costs=train_fun(xtrain_batch,ytrain_batch)
    accu=acc_fun(xtest,ytest)
    print "cost in epoch", i, "acc", accu
