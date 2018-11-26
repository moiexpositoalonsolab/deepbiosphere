import pandas as pd
import os
import matplotlib.pyplot as plt
#import ee
#import ee.mapclient
#ee.Initialize()
from PIL import Image
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision.transforms as transforms
import math
from os import listdir
from os.path import isfile, join

from DEEPBIO import *
from EEBIO import *


# Seed
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

################################################################################
## Import image
################################################################################

# Read all images under sta folder
ima=readsatelliteimages('../sat')
#fi='../sat/1deg_36dot6_-122.B10.tif'

# design subsampling strategy
lon=36.6
lat=-122
step=1
pixside=50
breaks=int(ima.shape[0]/pixside)


################################################################################
### Read gbif dataset and make the label tensor
################################################################################
from UTILS import *

# read gbif dataset
d = pd.read_csv("../gbif/pgbif.csv",sep="\t")
d.head()
print(d.size)

# Subset to SF
d_ = subcoor(d,lon,lat)

print(d_.head())

#Generate square grid from axis
spp=makespphash(d.iloc[:,0])
spptot=len(spp)
sppdic=make_sppdic(spp,spptot)

#tens=maketensor(10,10,spptot) # this for future implementation
tens=maketensor(2,breaks+1,breaks+1)#only for cactaceae

# generate translators of location
londic=make_locdic(lon,breaks+1)
latdic=make_locdic(lat,breaks+1)

# total cactus
# d_[d_['family']=='Cactaceae'].size
d_[d_['family']=='Brassicaceae'].size

# fill the tensor

sb= step/breaks
xwind=[[lon+(sb*i),lon+(sb*(i+1))]  for i in range(int(breaks))]
ywind=[[lat+(sb*i),lat+(sb*(i+1))]  for i in range(int(breaks))]
def whichwindow(w,v):
    count=0
    for i in w:
        if v>=i[0] and v<i[1]:
            break
        else:
            count=count+1
    return count
whichwindow(xwind,-118)
def iffamily(val,fam='Brassicaceae'):
    if val==fam:
        return(1)
    else:
        return(0)


for index, r in d_.iterrows():
#    print(r)
    da=whichwindow(ywind,r[1])
    do=whichwindow(xwind,r[2])
    dspp=iffamily(r[0],'Cactaceae')
    # da=key_for_value(latdic,round(r[1],1))
    # do=key_for_value(londic,round(r[2],1))
    #d3=key_for_value(sppdic,r[0]) # for future implementation
    # dspp=key_for_cactaceae(r[0])
    tens[dspp,da,do]= tens[dspp,da,do] +1

print(tens)

# total observation per grid
totobs=tens.sum(axis=0)

# % of cactaceae
cactae=tens[1,:,:]/(totobs+0.0001)
cactae=(cactae>0.01)*1
cactae

################################################################################
## setup Net and optimizers
################################################################################
#from DEEPBIO import *
#net=Net()
#print(net)

def outputSize(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    return(output)

## Fully connected NN
class Net(nn.Module):
    def __init__(self):
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
        #self.fc1 = nn.Linear(28 * 28, 200)
        self.fc1 = nn.Linear(in_features=50*50, out_features=200) # image dimensions 50
        self.fc2 = nn.Linear(200, 200)
        self.fc3 = nn.Linear(200, 1) # cactacea % | i can do 2 if cactacea/not | but perhaps output can be a tensor itself of 2*1
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # return F.log_softmax(x)
        # return F.softmax(x)
        return torch.sigmoid(x)

## Convoluted network

class Net(torch.nn.Module):
    #Our batch shape for input x is (3, 32, 32)
    def __init__(self):
        """
        https://blog.algorithmia.com/convolutional-neural-nets-in-pytorch/
        My input image is 1 chanel of 1x50x50
        This will go into a 18x50x50
        And to 18x25x25
        And this to a fully-connected layer of 60 nodes
        To output
        """
        super(Net, self).__init__()
        #Input channels = 3, output channels = 18
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=18, kernel_size=3, stride=1, padding=1)
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
        self.fc1 = torch.nn.Linear(18 * 25 * 25, 64) # 25 from  outputSize(50,2,2,0)
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
        x = x.view(-1, 18 * 25 *25)
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        x = F.relu(self.fc1(x))
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)
        x = self.fc2(x)
        return(x)

x = np.random.randn(32, D_in)
y = np.random.randn(N, D_out)
inputs=np.random.randn(3,32,32)


def myCrossEntropyLoss(outputs, labels):
    '''
    https://cs230-stanford.github.io/pytorch-getting-started.html
    '''
    batch_size = outputs.size()[0]            # batch_size
    outputs = F.log_softmax(outputs, dim=1)   # compute the log of softmax values
    outputs = outputs[range(batch_size), labels] # pick the values corresponding to the labels
    return -torch.sum(outputs)/num_examples

def accuracyrate(out, labels): # for classification
  outputs = np.argmax(out, axis=1)
  return np.sum(outputs==labels)/float(labels.size)
import operator
from functools import reduce
def prod(iterable):
    return reduce(operator.mul, iterable, 1)
def accuracy(x, y):
    x= x.detach().numpy()
    x.shape=prod(x.shape)
    y= y.detach().numpy()
    y.shape=prod(y.shape)
    r=np.corrcoef(x,y)
    return(r[0,1])


################################################################################
# Initialize
net=Net()
print(net)

momentum_=0.9
learning_rate=0.001
optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum_)
criterion = nn.MSELoss() # for continuous


def createLossAndOptimizer(net, learning_rate=0.001):
    #Loss function
    loss = torch.nn.CrossEntropyLoss()
    #Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    return(loss, optimizer)

loss, optimizer = createLossAndOptimizer(net, learning_rate)

################################################################################
## training process
################################################################################
#epochs=3
#for epoch in range(epochs):
totimages=100
batch_size=10
counter=0
running_loss = 0.0


wind=[[pixside*i,(pixside-1)+pixside*i] for i in range(int(breaks))]
wind=[[pixside*i,(pixside)+pixside*i] for i in range(int(breaks))]

### Example one set of inputs
# get the inputs
ys=np.random.choice(range(0, breaks-1), batch_size)
xs=np.random.choice(range(0, breaks-1), batch_size)

inputs=[[ima[wind[i][0]:wind[i][1],wind[j][0]:wind[j][1]]] for i,j in zip(ys,xs)] # the [] important to define dymensions
inputs=np.array(inputs, dtype='f')
inputs=torch.from_numpy(inputs)
inputs=inputs.view(-1, pixside*pixside) # for fully connected, reshape with view

# real
labels=[ cactae[i,j] for i,j in zip(ys,xs)]
# simulated
# labels=[ np.random.rand() for i,j in zip(ys,xs)]
labels=np.array(labels,dtype='f')
labels.shape=(10,1)
labels=torch.from_numpy(labels)

# zero the parameter gradients
optimizer.zero_grad()
# forward + backward + optimize
outputs = net(inputs)
# loss = np.square(outputs - labels).sum() ## manual, but does not work because needs gradient
loss = criterion(outputs, labels) ## done by package?
loss.backward()
optimizer.step()
acc=accuracy(outputs,labels)
# print progress
print('Train count: %i | Loss: %f | Accuracy: %f' %(counter,loss.data[0],acc))

################################################################################
## Real Training
totimages=100
batch_size=10
counter=0
running_loss = 0.0
n_epochs=10
# conv=True
conv=False

for epoch in range(n_epochs):
    running_loss = 0.0
    for i in range(int(totimages/batch_size)):
        # get the inputs
        ys=np.random.choice(range(0, breaks-1), batch_size)
        xs=np.random.choice(range(0, breaks-1), batch_size)
        # load inputs
        inputs=[[ima[wind[i][0]:wind[i][1],wind[j][0]:wind[j][1]]] for i,j in zip(ys,xs)] # the [] important to define dymensions
        inputs=np.array(inputs, dtype='f')
        inputs=torch.from_numpy(inputs)
        inputs=inputs.view(-1, pixside*pixside) # this is for fully connected
        # real labels
        labels=[ cactae[i,j] for i,j in zip(ys,xs)]
        labels=np.array(labels,dtype='f')
        labels.shape=(10,1)
        labels=torch.from_numpy(labels)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        # loss = np.square(outputs - labels).sum() ## manual, but does not work because needs gradient
        loss = criterion(outputs, labels) ## done by package?
        loss.backward()
        optimizer.step()
        # print progress
        running_loss += loss.data[0]
        acc=accuracy(outputs,labels)
        print('Train count: %i | Loss: %f | Accuracy: %f' %(counter,running_loss,acc))
        counter += 1


# for x in range(0,10):
#     for y in range(0,10):
#         # get the inputs
#         inputs=torch.from_numpy(ima[xc:10,yc:10])
#         # inputs=inputs..view(-1, 10*10)
#         inputs=inputs.contiguous().view(-1, 10*10)
#
#         labels=cactae[xc,yc]
#
#         # zero the parameter gradients
#         optimizer.zero_grad()
#
#         # forward + backward + optimize
#         outputs = net(inputs)
#         loss = criterion(outputs, labels)
#         loss.backward()
#         optimizer.step()
#
#         # print progress
#         print('Train count: %d | Loss: %d' %(counter,loss/data[0]))
#
#         loss.data[0]
#         # update counter y and total
#         yc=yc+1
#         counter=counter+1
#     # uppate counter x
#     xc=xc+1

#    # print statistics
#    running_loss += loss.item()
#    if i % 2000 == 1999:    # print every 2000 mini-batches
#        print('[%d, %5d] loss: %.3f' %
#              (epoch + 1, i + 1, running_loss / 2000))
#        running_loss = 0.0


################################################################################
## Compare with sklearn and SVMs
#Import the support vector machine module from the sklearn framework
from sklearn import svm

#Label x and y variables from our dataset
x = ourData.features
y = ourData.labels
#Initialize our algorithm

classifier = svm.SVC()

#Fit model to our data

################################################################################
### To save model
# https://cs230-stanford.github.io/pytorch-getting-started.html

# Painless Debugging
# With its clean and minimal design, PyTorch makes debugging a breeze. You can place breakpoints using
#pdb.set_trace() at any line in your code. You can then execute further computations, examine the PyTorch Tensors/Variables and pinpoint the root cause of the error.
