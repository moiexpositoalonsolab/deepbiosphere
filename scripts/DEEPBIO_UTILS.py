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
    def __init__(self,
                num_channels,
                pix_side,
                categories,
                net_type='cnn',
                optimal="ADAM",
                loss_fn="MSE",
                learning_rate=0.001,
                momentum=0.9):
        self.learning_rate=learning_rate
        self.num_channels=num_channels
        self.pix_side=pix_side
        self.net_type=net_type
        self.categories=categories
        self.optimal=optimal
        self.momentum=momentum
        self.loss_fn=loss_fn

def createLossAndOptimizer(net,par):
    #Loss function
    if par.loss_fn=="MSE" :
        loss = torch.nn.MSELoss() # for continuous
    elif par.loss_fn=="MUL":
        loss = torch.nn.MultiLabelMarginLoss()
    elif par.loss_fn=="ENT":
        loss = torch.nn.CrossEntropyLoss()
    else:
        loss = torch.nn.MSELoss() # for continuous
    #Optimizer
    if par.optimal=="ADAM" :
        optimizer = optim.Adam(net.parameters(), lr=par.learning_rate)

    elif par.optimal=="SDG":
        optimizer=optim.SGD(net.parameters(), lr=par.learning_rate,momentum=par.momentum)
    else:
        optimizer = optim.Adam(net.parameters(), lr=par.learning_rate)
    #
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

def addnoise(labels,v=0.001):
    noi=np.array(np.random.rand(prod(labels.shape))*v,dtype="f")
    noi.shape=labels.shape
    noi=torch.from_numpy(noi)
    return(labels+noi)

def isnan(x):
    return x != x

def subsetimagetensor(ima,z,y,x,net_type,channels, pix_side):
    breaks=ima.shape[len(ima.shape)-1]/pix_side
    wind=[[pix_side*i,(pix_side)+pix_side*i] for i in range(int(breaks))]
    inputs=[ ima[l, : , wind[i][0]:wind[i][1]  ,  wind[j][0]:wind[j][1] ]  for l,i,j in zip(z,y,x)] # the [] important to define dymensions
    inputs=np.array(inputs, dtype='f')
    inputs=torch.from_numpy(inputs)
    if(net_type=="fc"):
        inputs=inputs.view(-1, channels*pix_side*pix_side) # this is for fully connected
    return(inputs)

def subsetlabeltensor(spptensor,y,x,z,categories,batch_size):
    labels=[ spptensor[l,:, i,j] for l,i,j in zip(z,y,x)]
    labels=np.array(labels,dtype='f')
    labels.shape=(batch_size,categories)
    labels=torch.from_numpy(labels)
    return(labels)


def trainnet(ima, spptensor, ytrain,xtrain, net,par,loss,optimizer, epochs=10, reps=10, batch_size=10):
    running_loss = 0.0
    counter=0
    print('set\tepoch\trep\tcount\tloss\tr\n')
    for epoch in range(epochs):
        for n in range(reps):
                # get random inputs
                zs=np.random.choice(range(0,ima.shape[0]), batch_size)
                ys=np.random.choice(ytrain, batch_size)
                xs=np.random.choice(xtrain, batch_size)
                ###############################################
                # load inputs
                inputs=subsetimagetensor(ima,zs,ys,xs,net_type=par.net_type,channels=par.num_channels,pix_side=par.pix_side)
                ###############################################
                # real labels
                labels=subsetlabeltensor(spptensor,ys,xs,zs,spptensor.shape[1],batch_size)
                if labels.sum()==0:
                        print("No observations of taxon in this grid")
                        continue
                ###############################################
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs)
                loss_rec = loss(outputs, labels) ## done by package?
                if isnan(loss_rec):
                    loss_rec = loss(outputs, addnoise(labels))
                loss_rec.backward()
                optimizer.step()
                # print progress
                running_loss += loss_rec.item()
                acc=accuracy(outputs,labels)
                print('train\t{}\t{}\t{}\t{}\t{}\n' .format(epoch,n,counter,running_loss,acc))
                counter +=1
