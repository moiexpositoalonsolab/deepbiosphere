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

import pdb

# Seed
seed = 1
np.random.seed(seed)
torch.manual_seed(seed)

################################################################################
### Read gbif dataset and make the label tensor
################################################################################
from DEEPBIO_GBIF import *

spptensor=np.load("../gbif/gbiftensor.npy")
obsdensity=np.load("../gbif/gbifdensity.npy")
sppdic=np.load("../gbif/gbifdic.npy").item()
spptensor.sum(axis=0).sum(axis=1).sum(axis=1)

# spptensor=spptensor[:,1:6,:,:]
# spptensor=spptensor[:,5:7,:,:]
# spptensor=spptensor[:,0:50,:,:]
# spptensor=spptensor[:,10:12,:,:]
spptensor=spptensor[:,key_for_value(sppdic,"Cactaceae"):key_for_value(sppdic,"Cactaceae")+1,:,:]

# print("Taxon abundances")
spptensor.sum(axis=0).sum(axis=1).sum(axis=1)
spptensor.shape

# any manipulation?
#spptensor=spptensor*100
#spptensor = 1*(spptensor > 0.005)
#spptensor = (spptensor / spptensor.mean()) -1

categories=spptensor.shape[1]
breaks=spptensor.shape[3]-1

###############################################################################
## Import satellite images
################################################################################

ima=np.load("../satellite/rasters.npy")


numchannels=int(ima.shape[1])
totrasters=int(ima.shape[0])
pixside=int(ima.shape[2]/breaks)

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

globalclassimbalance = spptensor.sum()/prod(spptensor.shape)
weights=torch.from_numpy(np.array([globalclassimbalance], dtype="f"))
par=Params(
           num_channels=numchannels,
           pix_side=pixside,
           categories=categories,
           optimal='ADAM',
           loss_fn='BCE',
           loss_w=1/weights,
           net_type='cnn',
           learning_rate=0.01)

net=Net(par)
loss, optimizer = createLossAndOptimizer(net,par)


# Example prediction
batch_size=100
zs=np.random.choice(range(0,ima.shape[0]), batch_size)
ys=np.random.choice(ytrain, batch_size)
xs=np.random.choice(xtrain, batch_size)
inputs=subsetimagetensor(ima,zs,ys,xs,net_type=par.net_type,channels=par.num_channels,pix_side=par.pix_side)
labels=subsetlabeltensor(spptensor,ys,xs,zs,spptensor.shape[1],batch_size,datatype=tell_dtype_fromloss(par.loss_fn))
outputs = net(inputs)
outputs
labels
loss((outputs),labels)
accuracy(outputs,labels)

# train net
trainnet(ima=ima, spptensor=spptensor,ytrain=ytrain,xtrain=xtrain,ytest=ytest,xtest=xtest,
        net=net,par=par,loss=loss,optimizer=optimizer, epochs=10,batch_size=100)



#for epoch in range(args.n_epochs):
#    for n in range(args.n_replicates):
#        for m in range(totrasters):
#            # get random inputs
#            ys=np.random.choice(ytrain, batch_size)
#            xs=np.random.choice(xtrain, batch_size)
#            ###############################################
#            # load inputs
#            #    all channels  , window pos in lat   ,   window pos in lon
#            inputs=[ ima[m, : , wind[i][0]:wind[i][1]  ,  wind[j][0]:wind[j][1] ]  for i,j in zip(ys,xs)] # the [] important to define dymensions
#            inputs=np.array(inputs, dtype='f')
#            inputs=torch.from_numpy(inputs)
#            if(args.nn=="fc"):
#                inputs=inputs.view(-1, par.numchannels*par.pixside*par.pixside) # this is for fully connected
#            ###############################################
#            # real labels
#            labels=[ spptensor[m,i,j] for i,j in zip(ys,xs)]
#            labels=np.array(labels,dtype='f')
#            labels.shape=(batch_size,1)
#            labels=torch.from_numpy(labels)
#            if labels.sum()==0:
#                    print("No observations of taxon in this grid")
#                    continue
#            ###############################################
#            #pdb.set_trace()
#            # zero the parameter gradients
#            optimizer.zero_grad()
#            # forward + backward + optimize
#            outputs = net(inputs)
#            try:
#                loss_rec = loss(outputs, labels) ## done by package?
#            except:
#                loss_rec = loss(outputs, addnoise(labels))
#            if isnan(loss_rec):
#                loss_rec = loss(outputs, addnoise(labels))
#            loss_rec.backward()
#            optimizer.step()
#            # print progress
#            running_loss += loss_rec.item()
#            acc=accuracy(outputs,labels)
#            #f.write('train\t{}\t{}\t{}\t{}\t{}\t{}\n' .format(epoch,n,m,counter,running_loss,acc))
#            print('train\t{}\t{}\t{}\t{}\t{}\t{}\n' .format(epoch,n,m,counter,running_loss,acc))
#            ###############################################
#            ###############################################
#            # test
#            ys=np.random.choice(ytest, batch_size)
#            xs=np.random.choice(xtest, batch_size)
#            ###############################################
#            # load inputs
#            #    all channels  , window pos in lat   ,   window pos in lon
#            inputs=[ ima[m, : , wind[i][0]:wind[i][1]  ,  wind[j][0]:wind[j][1] ]  for i,j in zip(ys,xs)] # the [] important to define dymensions
#            inputs=np.array(inputs, dtype='f')
#            inputs=torch.from_numpy(inputs)
#            ###############################################
#            # real labels
#            labels=[ spptensor[m, i,j] for i,j in zip(ys,xs)]
#            labels=np.array(labels,dtype='f')
#            labels.shape=(batch_size,1)
#            labels=torch.from_numpy(labels)
#            # predict
#            outputs = net(inputs)
#            acc=accuracy(outputs,labels)
#            f.write('test\t{}\t{}\t{}\t{}\t{}\t{}\n' .format(epoch,n,m,counter,"-",acc))
#            print('test\t{}\t{}\t{}\t{}\t{}\t{}\n' .format(epoch,n,m,counter,"-",acc))
#            counter += 1


#f.close()

################################################################################
### To save model
# https://cs230-stanford.github.io/pytorch-getting-started.html

#PATH="../nets/cnn.tar"
#torch.save({
#            'epoch': epoch,
#            'model_state_dict': net.state_dict(),
#            'optimizer_state_dict': optimizer.state_dict(),
#            'loss': loss_rec,
#            }, PATH)
#
#modstored=torch.load(PATH)
#model=modstored["model_state_dic"]


#model.load_state_dict(torch.load(PATH))
#model.eval()
#model.train()

# Painless Debugging
# With its clean and minimal design, PyTorch makes debugging a breeze. You can place breakpoints using
#pdb.set_trace() at any line in your code. You can then execute further computations, examine the PyTorch Tensors/Variables and pinpoint the root cause of the error.
