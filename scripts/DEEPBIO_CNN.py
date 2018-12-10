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

# class Net(nn.Module):
#     def __init__(self,par):
#         self.pix_side=par.pix_side
#         self.categories=par.categories
#         self.num_channels=par.num_channels
#         #
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(self.num_channels, 64, 3,1,1)
#         self.conv2 = nn.Conv2d(64, 128, 3,1,1)
#         self.conv3 = nn.Conv2d(128, 256, 3,1,1)
#         self.pool = nn.MaxPool2d(2, 2)
#         self.pool2 = nn.MaxPool2d(5, 5)
#         self.fc1 = nn.Linear(256 * 5 * 5, 120)
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, self.categories)
#     def forward(self, x):
#         x = self.pool(F.relu(self.conv1(x)))
#         x = self.pool(F.relu(self.conv2(x)))
#         x = self.pool2(F.relu(self.conv3(x)))
#         x = x.view(-1, 256 * 5 * 5)
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return(torch.sigmoid(x))


class Net(nn.Module):
    def __init__(self,par):
        self.pix_side=par.pix_side
        self.categories=par.categories
        self.num_channels=par.num_channels
        #
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(self.num_channels, 64, 3,1,1)
        self.conv2 = nn.Conv2d(64, 128, 3,1,1)
        self.pool = nn.MaxPool2d(5, 5)
        self.fc1 = nn.Linear(128 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, self.categories)
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return(torch.sigmoid(x))

# Convoluted network
#class Net(torch.nn.Module):
#    #Our batch shape for input x is (3, 32, 32)
#    def __init__(self,par):
#        """
#        more info here:
#        https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/
#        https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-beginner-blitz-cifar10-tutorial-py
#        """
#        super(Net, self).__init__()
#        """ define parameters """
#        self.num_channels = par.num_channels
#        self.pix_side = par.pix_side
#        self.categories=par.categories
#        self.filters = 64 # convoluted layers, number of filters or depth
#        self.fchl = 120 # number of fully connected layers
#        """ define layers """
#        self.conv1 = torch.nn.Conv2d(in_channels=self.num_channels, out_channels=self.filters,2,1,0 )
#        self.pool = torch.nn.MaxPool2d(2,2)
#        self.fc1 = torch.nn.Linear(self.filters*25*25, self.fchl)
#        self.fc2 = torch.nn.Linear(self.fchl, self.categories)
#    def forward(self, x):
#        """ define forward connections """
#        x = F.relu(self.conv1(x))
#        x = self.pool(x)
#        bs, c, w, h = x.size()
#        x = x.view(-1, self.filters * self.out2_channels * self.out2_channels)
#        x = F.relu(self.fc1(x))
#        x = self.fc2(x)
##        return(x) # outputs any number
#        return(F.sigmoid(x))


#class Net(nn.Module):
#    def __init__(self):
#        super(Net, self).__init__()
#        self.conv1 = nn.Conv2d(30, 6, 3) # chanhged 3 for 30
#        self.pool = nn.MaxPool2d(2, 2)
#        self.conv2 = nn.Conv2d(6, 50, 3) # changed 16 for 50
#        self.fc1 = nn.Linear(50 * 3 * 3, 120) # changed 16 for 50
#        self.fc2 = nn.Linear(120, 84)
#        self.fc3 = nn.Linear(84, 5) # changed 10 for 5
#    def forward(self, x):
#        x = self.pool(F.relu(self.conv1(x)))
#        x = self.pool(F.relu(self.conv2(x)))
#        x = x.view(-1, 50 * 3 * 3)# changed 16 for 50
#        x = F.relu(self.fc1(x))
#        x = F.relu(self.fc2(x))
#        x = self.fc3(x)
#        return x
#
#
#### Convoluted network
#class Net(torch.nn.Module):
#    #Our batch shape for input x is (3, 32, 32)
#    def __init__(self, params):
#        """
#        Intro tutorial from https://github.com/cs230-stanford/cs230-code-examples
#        We define an convolutional network that predicts the sign from an image. The components
#        required are:
#        - an embedding layer: this layer maps each index in range(params.vocab_size) to a params.embedding_dim vector
#        - lstm: applying the LSTM on the sequential input returns an output for each token in the sentence
#        - fc: a fully connected layer that converts the LSTM output for each token to a distribution over NER tags
#        Args:
#            params: (Params) contains num_channels
#        """
#        super(Net, self).__init__()
#        self.num_channels = params.num_channels
#        self.side=params.pix_side
#        self.categories = params.categories
#        self.dropout_rate = 0.8
#        # each of the convolution layers below have the arguments (input_channels, output_channels, filter_size,
#        # stride, padding). We also include batch normalisation layers that help stabilise training.
#        # For more details on how to use these layers, check out the documentation.
#        self.conv1 = nn.Conv2d(self.num_channels, self.side, kernel_size=3, stride=1, padding=1)
#        self.bn1 = nn.BatchNorm2d(self.side)
#        self.conv2 = nn.Conv2d(self.side, self.side*2, kernel_size=3, stride=1, padding=1)
#        self.bn2 = nn.BatchNorm2d(self.side*2)
#        self.conv3 = nn.Conv2d(self.side*2, self.side*4, kernel_size=3, stride=1, padding=1)
#        self.bn3 = nn.BatchNorm2d(self.side*4)
#        # 2 fully connected layers to transform the output of the convolution layers to the final output
#        self.fc1 = nn.Linear(8*8*self.side*4, self.side*4)
#        self.fcbn1 = nn.BatchNorm1d(self.side*4)
#        self.fc2 = nn.Linear(self.side*4, self.categories)
#    def forward(self, s):
#        """
#        This function defines how we use the components of our network to operate on an input batch.
#        Args:
#            s: (Variable) contains a batch of images, of dimension batch_size x 3 x 64 x 64 .
#        Returns:
#            out: (Variable) dimension batch_size x 6 with the log probabilities for the labels of each image.
#        Note: the dimensions after each step are provided
#        """
#        #                                                  -> batch_size x 3 x 64 x 64
#        # we apply the convolution layers, followed by batch normalisation, maxpool and relu x 3
#        s = self.bn1(self.conv1(s))                         # batch_size x num_channels x 64 x 64
#        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels x 32 x 32
#        s = self.bn2(self.conv2(s))                         # batch_size x num_channels*2 x 32 x 32
#        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*2 x 16 x 16
#        s = self.bn3(self.conv3(s))                         # batch_size x num_channels*4 x 16 x 16
#        s = F.relu(F.max_pool2d(s, 2))                      # batch_size x num_channels*4 x 8 x 8
#        # flatten the output for each image
#        s = s.view(-1, 8*8*self.self.side*4)             # batch_size x 8*8*num_channels*4
#        # apply 2 fully connected layers with dropout
#        s = F.dropout(F.relu(self.fcbn1(self.fc1(s))),
#            p=self.dropout_rate, training=self.training)    # batch_size x self.num_channels*4
#        s = self.fc2(s)                                     # batch_size x 6
#        # apply log softmax on each image's output (this is recommended over applying softmax
#        # since it is numerically more stable)
#        # return F.log_softmax(s, dim=1)
#        return F.sigmoid(s, dim=1) # I prefer sigmoid
#
#
# def loss_fn(outputs, labels):
#     """
#     Compute the cross entropy loss given outputs and labels.
#     Args:
#         outputs: (Variable) dimension batch_size x 6 - output of the model
#         labels: (Variable) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
#     Returns:
#         loss (Variable): cross entropy loss for all images in the batch
#     Note: you may use a standard loss function from http://pytorch.org/docs/master/nn.html#loss-functions. This example
#           demonstrates how you can easily define a custom loss function.
#     """
#     num_examples = outputs.size()[0]
#     return -torch.sum(outputs[range(num_examples), labels])/num_examples
#
#
# def accuracy(outputs, labels):
#     """
#     Compute the accuracy, given the outputs and labels for all images.
#     Args:
#         outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
#         labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]
#     Returns: (float) accuracy in [0,1]
#     """
#     outputs = np.argmax(outputs, axis=1)
#     return np.sum(outputs==labels)/float(labels.size)
#
# # maintain all metrics required in this dictionary- these are used in the training and evaluation loops
# metrics = {
#     'accuracy': accuracy,
#     # could add more metrics such as accuracy for each token type
# }
