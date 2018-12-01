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


## Fully connected NN
class Net(nn.Module):
    def __init__(self,par):
        """
        https://cs230-stanford.github.io/pytorch-getting-started.html
        https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
        Layer 1: Dimensions input = total pixels in an image, e.g. 50x50
        Dimensions of hidenn layers are 200
        Layer 2: 200 to 200 nodes
        Later 3: 200 nodes
        to the output dimension, which is a number, 1
        """
        super(Net, self).__init__()
        self.num_channels = par.num_channels
        self.categories = par.categories
        self.pix_side = par.pix_side
        self.hid = 200
        self.fc1 = nn.Linear(in_features=par.num_channels*self.pix_side*self.pix_side, out_features=self.hid) # image dimensions 50
        self.fc2 = nn.Linear(self.hid, self.hid)
        self.fc3 = nn.Linear(self.hid, self.categories) # cactacea % | i can do 2 if cactacea/not | but perhaps output can be a tensor itself of 2*1
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # return F.log_softmax(x)
        # return F.softmax(x)
        return torch.sigmoid(x)
