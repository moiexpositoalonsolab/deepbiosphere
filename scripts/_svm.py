"""
Train a Deep Neural Network with Biodiversity labels
@author: moisesexpositoalonso@gmail.com
"""

import os
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import torchvision.transforms as transforms

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device: {}".format(device))

# Seed
seed = 1
np.random.seed(seed);
torch.manual_seed(seed);

#os.chdir("/ebio/abt6/mexposito/ebio/abt6_projects7/ath_1001G_field/deepbiosphere/scripts")

################################################################################
### Read gbif dataset and make the label tensor
################################################################################
from UTILS import *

spptensor=np.load("../gbif/gbiftensor.npy")
obsdensity=np.load("../gbif/gbifdensity.npy")
sppdic=np.load("../gbif/gbifdic.npy").item()
sppc=spptensor.sum(axis=0).sum(axis=1).sum(axis=1)

allspp=True
if (allspp):
    sppt=np.array([spptensor[:,i,:,:] for i in range(len(sppc)) if sppc[i] >10])
    sppt=sppt.transpose(1,0,2,3)
    spptensor=sppt
else:
    spptensor=np.arraY(spptensor[:,key_for_value(sppdic,"Cactaceae"):key_for_value(sppdic,"Cactaceae")+1,:,:])

spptensor.sum(axis=0).sum(axis=1).sum(axis=1)
spptensor.shape

spptensor=torch.from_numpy(spptensor)
spptensor=spptensor.type('torch.FloatTensor')


categories=spptensor.shape[1]
breaks=spptensor.shape[3]-1

###############################################################################
## Import satellite images
################################################################################

ima=np.load("../satellite/rasters.npy")

# breakdown images
numchannels=int(ima.shape[1]);
totrasters=int(ima.shape[0])
pixside=int(ima.shape[2]/breaks)
wind=[[pixside*i,(pixside)+pixside*i] for i in range(int(breaks))]
ima=np.array([[ima[:,:,wind[x][0]:wind[x][1],wind[y][0]:wind[y][1]] for y in range(int(breaks))] for x in range(int(breaks))])
ima=torch.Tensor(ima)


# Define sampling of images
ytrain=np.random.choice(range(0, breaks), int(round(breaks * 0.6,1)) ,replace=False)
xtrain=np.random.choice(range(0, breaks), int(round(breaks * 0.6,1)) ,replace=False)
ytest=[i for i in range(0, breaks) if i not in ytrain ]
xtest=[i for i in range(0, breaks) if i not in xtrain ]



# ytrain=range(0, breaks-1)
# xtrain=range(0, breaks-1) # until debugging

### Check they are aligned
f1 = [line.rstrip('\n') for line in open("../satellite/rasters.info","r")]
f2 = [line.rstrip('\n') for line in open("../gbif/gbiftensor.info","r")]
if f1 != f2:
   raise Exception("The species and imagery grids do not seem aligned")


################################################################################
## setup Net and optimizers and train
################################################################################
import DEEPBIO_CNN
import DEEPBIO_UTILS
from DEEPBIO_UTILS import *
from DEEPBIO_CNN import *


# Example prediction
batch_size=500
zs=np.array(range(0,ima.shape[0])).tolist()
training=np.array([[[[x,y,z] for y in ytrain] for x in xtrain ] for z in zs])
training.shape=((training.shape[0]*training.shape[1]*training.shape[2] ), 3)
training=np.transpose(training,(1,0))
xs=training[0].tolist()
ys=training[1].tolist()
zs=training[2].tolist()
inputs=subsetimagetensor(ima,zs,ys,xs,net_type=par.net_type,channels=par.num_channels,pix_side=par.pix_side)
# for non-convolutional approaches, inputs need to be reduced. I use mean
inputs=inputs.mean(dim=[2,3])
inputs_df=pd.DataFrame(inputs.numpy())
labels=subsetlabeltensor(spptensor,ys,xs,zs,spptensor.shape[1],batch_size,datatype=tell_dtype_fromloss(par.loss_fn))

# test data
batch_size=500
zs=np.array(range(0,ima.shape[0])).tolist()
testing=np.array([[[[x,y,z] for y in ytest] for x in xtest ] for z in zs])
testing.shape=((testing.shape[0]*testing.shape[1]*testing.shape[2] ), 3)
testing=np.transpose(testing,(1,0))
xs=testing[0].tolist()
ys=testing[1].tolist()
zs=testing[2].tolist()
tinputs=subsetimagetensor(ima,zs,ys,xs,net_type=par.net_type,channels=par.num_channels,pix_side=par.pix_side)
# for non-convolutional approaches, inputs need to be reduced. I use mean
tinputs=tinputs.mean(dim=[2,3])
tinputs_df=pd.DataFrame(tinputs.numpy())
tlabels=subsetlabeltensor(spptensor,ys,xs,zs,spptensor.shape[1],batch_size,datatype=tell_dtype_fromloss(par.loss_fn))



#Import the support vector machine module from the sklearn framework
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor

rf = RandomForestRegressor(n_estimators = 1000, random_state = 42)
rf.fit(inputs_df, labels[:,0]);
rf.fit(inputs_df, labels);

outputs=rf.predict(tinputs_df)

accuracy(torch.Tensor(outputs),tlabels)
precision(torch.Tensor(outputs),tlabels)
recall(torch.Tensor(outputs),tlabels)


accuracy(torch.Tensor(outputs),tlabels[:,0])
precision(torch.Tensor(outputs),tlabels[:,0])
recall(torch.Tensor(outputs),tlabels[:,0])




