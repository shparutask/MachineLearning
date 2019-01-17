from time import time, sleep
import numpy as np
import matplotlib.pyplot as plt

import Sequential as seq

import Linear as linear
import Soft_Max as softMax

import ReLU as r
import Soft_Plus as softplus

import MSECriterion as mse
import ClassNLLCriterionUnstable as nllu

import Helper as help

# Generate some data
N = 500

X1 = np.random.randn(N,2) + np.array([2,2])
X2 = np.random.randn(N,2) + np.array([-2,-2])

Y = np.concatenate([np.ones(N),np.zeros(N)])[:,None]
Y = np.hstack([Y, 1-Y])

X = np.vstack([X1,X2])
plt.scatter(X[:,0],X[:,1], c = Y[:,0], edgecolors= 'none')

#Define a logistic regression for debugging
net = seq.Sequential()
net.add(linear.Linear(2, 2))
net.add(softMax.SoftMax())

criterion = mse.MSECriterion()

# Test something like that then 

#net = seq.Sequential()
#net.add(linear.Linear(2, 4))
#net.add(r.ReLU())
#net.add(linear.Linear(4, 2))
#net.add(softplus.SoftPlus())

#criterion = nllu.ClassNLLCriterionUnstable()

print(net)

# Looping params
n_epoch = 20
batch_size = 128

help.run_network(X, Y, net, criterion, n_epoch, batch_size)

