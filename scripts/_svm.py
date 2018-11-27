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
lon=36.6
lat=-122
step=1
pixside=50
imagesize=500
breaks=int(500/pixside)



################################################################################
## Import satellite images
################################################################################
from EEBIO import *

# Read all images under sta folder
ima=readsatelliteimages('../sat')
#fi='../sat/1deg_36dot6_-122.B10.tif'

################################################################################
### Read gbif dataset and make the label tensor
################################################################################
from GBIF import *

# read gbif dataset
d=readgbif(path="../gbif/pgbif.csv")

# make species map
biogrid=tensoronetaxon(step, breaks, lon,lat, d, "Lauraceae")


################################################################################
## setup Net and optimizers
################################################################################
from DEEPBIO import *
import DEEPBIO

################################################################################
## training
################################################################################

totimages=100
batch_size=10
counter=0
running_loss = 0.0
n_epochs=10

wind=[[pixside*i,(pixside)+pixside*i] for i in range(int(breaks))]

ytrain=np.random.choice(range(0, breaks-1), int(round(breaks * 0.7,1)) ,replace=False)
xtrain=np.random.choice(range(0, breaks-1), int(round(breaks * 0.7,1)) ,replace=False)

ytest=[i for i in range(0, breaks-1) if i not in ytrain ]
xtest=[i for i in range(0, breaks-1) if i not in xtrain ]

for epoch in range(n_epochs):
    running_loss = 0.0
    for i in range(int(totimages/batch_size)):
        # get random inputs
        ys=np.random.choice(ytrain, batch_size)
        xs=np.random.choice(xtrain, batch_size)
        ###############################################
        # load inputs
        #    all channels  , window pos in lat   ,   window pos in lon
        inputs=[ ima[: , wind[i][0]:wind[i][1]  ,  wind[j][0]:wind[j][1] ]  for i,j in zip(ys,xs)] # the [] important to define dymensions
        inputs=np.array(inputs, dtype='f')
        inputs=torch.from_numpy(inputs)
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
        running_loss += loss_rec.data[0]
        acc=accuracy(outputs,labels)
        print('Train count: %i | Loss: %f | Accuracy: %f' %(counter,running_loss,acc))
        counter += 1



## Compare with sklearn and SVMs
################################################################################
#Import the support vector machine module from the sklearn framework
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor

# testing
ys=np.random.choice(ytrain, batch_size)
xs=np.random.choice(xtrain, batch_size)
ys=ytrain
xs=xtrain

#Label x and y variables from our dataset
inputs=inputs=[ ima[: , wind[i][0]:wind[i][1]  ,  wind[j][0]:wind[j][1] ]  for i,j in zip(ys,xs)] # 
inputs=np.array(inputs, dtype='f')
inputs_mid=pd.DataFrame( inputs[:,:,25,25]  )
inputs_df=pd.DataFrame(inputs_mid)

labels=[ biogrid[i,j] for i,j in zip(ys,xs)]
labels_df=pd.DataFrame(labels)
rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(inputs_df, labels);

acc = np.corrcoef( rf.predict(inputs_mid) , labels)[1,0]
print("Accuracy\t{}" .format(acc)  )


#test
ys=np.random.choice(ytest, batch_size)
xs=np.random.choice(xtest, batch_size)

inputs=inputs=[ ima[: , wind[i][0]:wind[i][1]  ,  wind[j][0]:wind[j][1] ]  for i,j in zip(ys,xs)] #
inputs=np.array(inputs, dtype='f')
inputs_mid=pd.DataFrame( inputs[:,:,25,25]  )

labels=[ biogrid[i,j] for i,j in zip(ys,xs)]


acc = np.corrcoef( rf.predict(inputs_mid) , labels)[1,0]
print("Accuracy\t{}" .format(acc)  )




