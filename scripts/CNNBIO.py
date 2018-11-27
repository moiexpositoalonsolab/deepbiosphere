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

## Convoluted network
class Net(torch.nn.Module):
    #Our batch shape for input x is (3, 32, 32)
    def __init__(self,par):
        """
        https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/
        My input image is 1 chanel of 1x50x50
        This will go into a 18x50x50
        And to 18x25x25
        And this to a fully-connected layer of 60 nodes
        To output
        """
        super(Net, self).__init__()
        self.num_channels = par.num_channels
        self.pix_side = par.pix_side
        self.out_channels= outputSize(self.pix_side,2,2,0)
        self.conv = 18
        #Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(in_channels=self.num_channels, out_channels=self.conv, kernel_size=3, stride=1, padding=1)
            # Kernel Size – the size of the filter.
            # Kernel Type – the values of the actual filter. Some examples include identity, edge detection, and sharpen.
            # Stride – the rate at which the kernel passes over the input image. A stride of 2 moves the kernel in 2 pixel increments.
            # Padding – we can add layers of 0s to the outside of the image in order to make sure that the kernel properly passes over the edges of the image.
            # Output Layers – how many different kernels are applied to the image.
            # kernel 3 stride 1 padding 1 does not change dymensions of imageself.
            # we want to nevertheless transform an image from 1 channel into 18 chanels (i.e. abstracting the image)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #4608 input features, 64 output features (see sizing flow below)
        # self.fc1 = torch.nn.Linear(18 * 16 * 16, 64)
        self.fc1 = torch.nn.Linear(self.conv * self.out_channels * self.out_channels, 64) # 25 from  outputSize(50,2,2,0)
        #64 input features, 10 output features for our 10 defined classes
        self.fc2 = torch.nn.Linear(64, 1)
    def forward(self, x):
        #Computes the activation of the first convolution
        #Size changes from (3, 32, 32) to (18, 32, 32)
        x = F.relu(self.conv1(x))
        #Size changes from (18, 32, 32) to (18, 16, 16)
        x = self.pool(x)
        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 16, 16) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension
        # x = x.view(-1, 18 * 16 *16) # also needed to change
        x = x.view(-1, 18 * self.out_channels * self.out_channels)
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return(x)
