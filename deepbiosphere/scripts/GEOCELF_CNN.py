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
def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    return(output)


class Net(nn.Module):
    """
    Checking - it requires more training time, 1 layer more 
    """
    def __init__(self, categories, num_channels):
#         self.pix_side=par.pix_side

        super(Net, self).__init__()
        self.categories=categories
        self.num_channels=num_channels
#         self.kernel = kernel
        self.conv1 = nn.Conv2d(self.num_channels, 64, 7,1,1) # try a kernel of size 7 like TNN model
        self.conv2 = nn.Conv2d(64, 128, 3,1,1)
        self.conv3 = nn.Conv2d(128, 256, 3,1,1)
        self.conv4 = nn.Conv2d(256, 256, 3,1,1)        
        self.conv5 = nn.Conv2d(256, 512, 3,1,1)        
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool5 = nn.MaxPool2d(5, 5)
        self.famfc = nn.Linear(256*6*6, 1858) # 1858 comes from # families in us dataset 
        #TODO: remove magic numbers & make constants
        self.genfc = nn.Linear(1858, 8668) #TODO: figure out what the heck these constants are.  Think they're from taxonomy of the different plant species?
        self.specfc = nn.Linear(8668, self.categories) 
        
        
    def forward(self, x): 
        x = self.pool2(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool5(x)
        #x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.famfc(x))
        x = F.relu(self.genfc(x))
        x = self.specfc(x)
        return(x)
#         return(torch.sigmoid(x))

# class Params():
#     """
#     Store hyperparameters to define Net
#     """
#     def __init__(self,
#                 num_channels,
#                 pix_side,
#                 categories,
#                 net_type='cnn',
#                 optimal="ADAM",
#                 loss_fn="MSE",
#                 loss_w=None,
#                 learning_rate=0.001,
#                 momentum=0.9):
#         self.learning_rate=learning_rate
#         self.num_channels=num_channels
#         self.pix_side=pix_side
#         self.net_type=net_type
#         self.categories=categories
#         self.optimal=optimal
#         self.momentum=momentum
#         self.loss_fn=loss_fn
#         self.loss_w=loss_w   
