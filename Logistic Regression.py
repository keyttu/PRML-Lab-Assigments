#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 11:14:29 2020

@author: akshay and Keyttu
"""

import numpy as np 
import struct
import matplotlib.pyplot as plt


'''
Reads the file with the following format
 TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000803(2051) magic number
0004     32 bit integer  60000            number of images
0008     32 bit integer  28               number of rows
0012     32 bit integer  28               number of columns
0016     unsigned byte   ??               pixel
0017     unsigned byte   ??               pixel
........
xxxx     unsigned byte   ??               pixel
'''
def read_image_file(fname):
   fin = open(fname,'rb');
   mnum =  struct.unpack('>i',fin.read(4)) 
   nimages =  struct.unpack('>i',fin.read(4)) 
   nrows =   struct.unpack('>i',fin.read(4)) 
   ncols =   struct.unpack('>i',fin.read(4)) 
   dim = nrows[0]*ncols[0] ;
   x = np.array(list(fin.read()));
   x = x.reshape(nimages[0],dim);
   fin.close()
   return x;

'''
TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
[offset] [type]          [value]          [description]
0000     32 bit integer  0x00000801(2049) magic number (MSB first)
0004     32 bit integer  60000            number of items
0008     unsigned byte   ??               label
0009     unsigned byte   ??               label
........
xxxx     unsigned byte   ??               label

The labels values are 0 to 9.
'''
def read_label_file(fname):
   fin = open(fname,'rb');
   mnum =  struct.unpack('>i',fin.read(4))
   nimages =  struct.unpack('>i',fin.read(4))  
   x = np.array(list(fin.read()));
   x = x.reshape(nimages[0],1);
   fin.close()
   return x;


X = read_image_file('train-images-idx3-ubyte');
Y = read_label_file('train-labels-idx1-ubyte');

print(X.shape)
print(Y.shape)


#######################
# display one image # 

print(Y[7,:]);
x = X[7,:].reshape(28,28);

plt.imshow(x.astype('uint8'));
plt.show()

'''
 Task 1 
 Learn a logistic regression classifier to 
 distinguish number three from rest of the classes 
'''

# Make a new array with `1' label for class 3 and `0' 
# label for other classes
y_new = np.zeros(60000).reshape(60000,1)
for i in range(60000):
    if(Y[i]==3):
        y_new[i] =1 


plt.hist(y_new)
plt.show()

X = X/255

# Initialize the weights, bias
#W = np.random.rand(784).reshape(784,1)/1 = 0
#B = np.random.rand()
W = np.zeros(784).reshape(784,1)
B = np.zeros(1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def LossFunction(y_pred,y_new):
    N = len(y_pred)
    temp = np.log(y_pred)
    temp1 = np.log(1-y_pred)
    temp2 = y_new*temp
    temp3 = (1-y_new)*temp1
    temp4 = temp2+temp3
    temp4 = (-1/N)*temp4
    return temp4.sum()

def GradientW(y_pred, y_new, X):
    N = len(y_pred)
    

def GradientB(y_pred, y_new, X):
    
# Compute the Cross Entropy 

# (consider writing a function)
# Print it 

alpha = 0.1
## Implement gradient descent 
for i in range(100): # start with 100 epochs 
    print('Epoch Number : 'i); # print epoch number
    y_pred = sigmoid(X.dot(W)+B)
    J = LossFunction(y_pred, y_new)
    print('The Loss is ',J)
    # Compute gradients
    dJ = GradientW(y_pred,y_new, X)
    dB = GradientB(y_pred, y_new)
    
    W = W - alpha*dJ
    B = B - alpha*dB
    
    # update wey_preights

# Reshape the weight into a 28x28 image 
# and do imshow 

## Perform testing 


## Find the accuracy on test data 

'''
 Task 2 
 Learn a logistic regression classifier for 
 10 class classification  ## Find the gradient updates 
'''


