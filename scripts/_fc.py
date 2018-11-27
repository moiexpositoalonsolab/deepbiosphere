import pandas as pd
import os
from os import listdir
from os.path import isfile, join

from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision.transforms as transforms
import math


# Seed
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

# Analysis design
lat=36.6
lon=-122
step=1
pixside=50
imagesize=500
breaks=int(500/pixside)
channels=28


################################################################################
## Import satellite images
################################################################################
from EEBIO import *

# Read all images under sta folder
ima=readsatelliteimages('../satellite')
#fi='../sat/1deg_36dot6_-122.B10.tif'

################################################################################
### Read gbif dataset and make the label tensor
################################################################################
from GBIF import *

# read gbif dataset
d=readgbif(path="../gbif/pgbif.csv")

# make species map
biogrid=tensoronetaxon(step, breaks, lat,lon, d, 'Lauraceae')
# biogrid=tensorgbif(step, breaks, lat,lon, d) ## for many classes
#print(biogrid)

################################################################################
## setup Net and optimizers
################################################################################
from DEEPUTILS import *
from FCBIO import *

par=Params(learningrate=0.001,
       numchannels=ima.shape[0],
       pixside=pixside)

net=Net(par)
loss, optimizer = createLossAndOptimizer(net, par.learning_rate)

#print(net)

################################################################################
## training
################################################################################

batch_size=10
n_epochs=5
n_reps=5

wind=[[pixside*i,(pixside)+pixside*i] for i in range(int(breaks))]

ytrain=np.random.choice(range(0, breaks-1), int(round(breaks * 0.7,1)) ,replace=False)
xtrain=np.random.choice(range(0, breaks-1), int(round(breaks * 0.7,1)) ,replace=False)

ytest=[i for i in range(0, breaks-1) if i not in ytrain ]
xtest=[i for i in range(0, breaks-1) if i not in xtrain ]

biogrid[ytrain,:]
biogrid[:,xtrain]
biogrid[ytest,:]
biogrid[:,xtest]


counter=0
running_loss = 0.0
print('run\tnumber\tloss\taccuracy')
for epoch in range(n_epochs):
    for i in range(n_reps):
        # get random inputs
        ys=np.random.choice(ytrain, batch_size)
        xs=np.random.choice(xtrain, batch_size)
        ###############################################
        # load inputs
        #    all channels  , window pos in lat   ,   window pos in lon
        inputs=[ ima[: , wind[i][0]:wind[i][1]  ,  wind[j][0]:wind[j][1] ]  for i,j in zip(ys,xs)] # the [] important to define dymensions
        inputs=np.array(inputs, dtype='f')
        inputs=torch.from_numpy(inputs)
        inputs=inputs.view(-1, channels*pixside*pixside) # this is for fully connected
        ###############################################
        # real labels
        labels=[ biogrid[i,j] for i,j in zip(ys,xs)]
        labels=np.array(labels,dtype='f')
        labels.shape=(10,1)
        labels=torch.from_numpy(labels)
        ###############################################
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = net(inputs)
        # loss_rec = np.square(outputs - labels).sum() ## manual, but does not work because needs gradient
        loss_rec = loss(outputs, labels) ## done by package?
        loss_rec.backward()
        optimizer.step()
        # print progress
        running_loss += loss_rec.item()
        acc=accuracy(outputs,labels)
        print('run\t%i\t%f\t%f' %(counter,running_loss,acc))
        counter += 1



################################################################################
### To save model
# https://cs230-stanford.github.io/pytorch-getting-started.html

PATH="../nets/fc.tar"
torch.save({
            'epoch': epoch,
            'model_state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss_rec,
            }, PATH)

modstored=torch.load(PATH)
model=modstored["model_state_dic"]

