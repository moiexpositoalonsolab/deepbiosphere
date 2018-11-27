"""
Defines Deep Neural Network implementations with torch for
Biodiversity Geo-Modeling

@author: moisesexpositoalonso@gmail.com
"""


import os
import numpy as np
import operator
from functools import reduce


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


###############################################################################
## Helpers
class Params():
    """
    Store hyperparameters to define Net
    """
    def __init__(self, learningrate,numchannels,pixside,nettype='cnn'):
        self.learning_rate=learningrate
        self.num_channels=numchannels
        self.pix_side=pixside
        self.net_type=nettype

def createLossAndOptimizer(net, learning_rate=0.001):
    #Loss function
    # loss = torch.nn.CrossEntropyLoss()
    loss = torch.nn.MSELoss() # for continuous
    #Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    return(loss, optimizer)


def successrate(out, labels): # for classification
  outputs = np.argmax(out, axis=1)
  return np.sum(outputs==labels)/float(labels.size)

def prod(iterable):
    return reduce(operator.mul, iterable, 1)

def accuracy(x, y):
    x= x.detach().numpy()
    x.shape=prod(x.shape)
    y= y.detach().numpy()
    y.shape=prod(y.shape)
    r=np.corrcoef(x,y)
    return(r[0,1])
