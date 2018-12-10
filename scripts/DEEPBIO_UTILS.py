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
                loss_w=None,
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
        self.loss_w=loss_w

def createLossAndOptimizer(net,par):
    #Loss function
    if par.loss_fn=="MSE" :
        loss = torch.nn.MSELoss() # for continuous
    elif par.loss_fn=="MUL":
        loss = torch.nn.MultiLabelMarginLoss()
    elif par.loss_fn=="BCE":
        if par.loss_w is not None:
            loss = torch.nn.BCELoss(weight=par.loss_w)
        else:
            loss = torch.nn.BCELoss()
    else:
        raise Exception("loss function name unknown! try MSE, MUL or BCE")
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

# def accuracy(x, y):
#     x= x.detach().numpy()
#     x.shape=prod(x.shape)
#     y= y.detach().numpy()
#     y.shape=prod(y.shape)
#     r=np.corrcoef(x,y)
#     return(r[0,1])

def accuracy(x, y):
    x=torch.round(x)
    a=(torch.sum(x==y)).double()/x.numel()
    return(a)


# def precision(o, l):
#     o= o.detach().numpy()
#     l= l.detach().numpy()
#     y.shape=prod(y.shape)
#     r=np.corrcoef(x,y)
#     return(r[0,1])
#
# def recall(x, y):
#     x= x.detach().numpy()
#     x.shape=prod(x.shape)
#     y= y.detach().numpy()
#     y.shape=prod(y.shape)
#     r=np.corrcoef(x,y)
#     return(r[0,1])

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

def subsetlabeltensor(spptensor,y,x,z,categories,batch_size,datatype='f'):
    labels=[ spptensor[l,:, i,j] for l,i,j in zip(z,y,x)]
    labels=np.array(labels,dtype=datatype)
    labels.shape=(batch_size,categories)
    labels=torch.from_numpy(labels)
    return(labels)

def outputs_datatype(outputs,lossname):
    if(lossname=='MUL' or lossname=='ENT'):
        outputs=torch.round(outputs) # CAREFUL WITH THIS
    return(outputs)

def tell_dtype_fromloss (lossname):
    if(lossname=='MSE'):
        datatype='f'
    elif(lossname=='BCE'):
        datatype='f'
    elif(lossname=='MUL'):
        datatype='l'
    else:
        datatype='f'
    return(datatype)

def trainnet(ima, spptensor, ytrain,xtrain,ytest,xtest, net,par,loss,optimizer, epochs=10, batch_size=10):
    running_loss = 0.0
    counter=0
    reps=round(spptensor.shape[0]*spptensor.shape[2]*spptensor.shape[3]/batch_size )
    print('line\tepoch\trep\tcount\tloss\tr_train\tr_test\n')
    for epoch in range(epochs):
        for n in range(reps):
                ###############################################
                # get random inputs
                zs=np.random.choice(range(0,ima.shape[0]), batch_size)
                ys=np.random.choice(ytrain, batch_size)
                xs=np.random.choice(xtrain, batch_size)
                # load inputs
                inputs=subsetimagetensor(ima,zs,ys,xs,net_type=par.net_type,channels=par.num_channels,pix_side=par.pix_side)
                # real labels
                labels=subsetlabeltensor(spptensor,ys,xs,zs,spptensor.shape[1],batch_size,datatype=tell_dtype_fromloss(par.loss_fn))
                if labels.sum()==0:
                        print("No observations of taxon in this grid")
                        continue
                ###############################################
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(inputs)
                # outputs = outputs_datatype(outputs,par.loss_fn)
                if par.loss_w is not None:
                    loss_rec = loss(outputs, labels,)
                else:
                    loss_rec = loss(outputs, labels)
                loss_rec.backward()
                optimizer.step()
                # print progress
                running_loss += loss_rec.item()
                acc=accuracy(outputs,labels)
                ###############################################
                # get random inputs
                zs=np.random.choice(range(0,ima.shape[0]), batch_size)
                ys=np.random.choice(ytest, batch_size)
                xs=np.random.choice(xtest, batch_size)
                # load inputs
                inputs=subsetimagetensor(ima,zs,ys,xs,net_type=par.net_type,channels=par.num_channels,pix_side=par.pix_side)
                # real labels
                labels=subsetlabeltensor(spptensor,ys,xs,zs,spptensor.shape[1],batch_size)
                outputs = net(inputs)
                acctest=accuracy(outputs,labels)
                ###############################################
                #out
                print('run\t{}\t{}\t{}\t{}\t{}\t{}\n' .format(epoch,n,counter,running_loss,acc,acctest))
                counter +=1
