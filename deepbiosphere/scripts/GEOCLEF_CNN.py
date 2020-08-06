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

class OGNet(nn.Module):
    """
    Checking - it requires more training time, 1 layer more 
    """
    def __init__(self, species, families, genuses, num_channels):

        super(OGNet, self).__init__()
        self.categories=species
        self.species = species
        self.families = families
        self.genuses = genuses
        self.num_channels=num_channels
        self.conv1 = nn.Conv2d(self.num_channels, 64, 7,1,1) # try a kernel of size 7 like TNN model
        self.conv2 = nn.Conv2d(64, 128, 3,1,1)
        self.conv3 = nn.Conv2d(128, 256, 3,1,1)
        self.conv4 = nn.Conv2d(256, 256, 3,1,1)        
        self.conv5 = nn.Conv2d(256, 512, 3,1,1)        
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool5 = nn.MaxPool2d(5, 5)
        self.famfc = nn.Linear(256*6*6, self.families) 
        self.genfc = nn.Linear(self.families, self.genuses)
        self.specfc = nn.Linear(self.genuses, self.species) 
        
        
    def forward(self, x): 
        x = self.pool2(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool5(x)
        #x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        fam = F.relu(self.famfc(x))
        gen = F.relu(self.genfc(fam))
        spec = self.specfc(gen)
        return(spec, gen, fam)

class SkipNet(nn.Module):
    """
    Checking - it requires more training time, 1 layer more 
    """
    def __init__(self, species, families, genuses, num_channels):

        super(SkipNet, self).__init__()
        self.categories=species
        self.species = species
        self.families = families
        self.genuses = genuses
        self.num_channels=num_channels
        self.conv1 = nn.Conv2d(self.num_channels, 64, 7,1,1) # try a kernel of size 7 like TNN model
        self.conv2 = nn.Conv2d(64, 128, 3,1,1)
        self.conv3 = nn.Conv2d(128, 256, 3,1,1)
        self.conv4 = nn.Conv2d(256, 256, 3,1,1)        
        self.conv5 = nn.Conv2d(256, 512, 3,1,1)        
        self.pool2 = nn.MaxPool2d(2, 2)
                               
        self.pool5 = nn.MaxPool2d(5, 5)
        self.chokepoint = 256*6*6
        self.famfc = nn.Linear(self.chokepoint, self.families) 
        # TODO: insert downsampling here???
        #self.fc1 = nn.Linear(256 * 5 * 5, 120)
        #self.fc2 = nn.Linear(120, 84)                               
        self.genfc = nn.Linear(self.chokepoint + self.families, self.genuses)
        self.specfc = nn.Linear(self.chokepoint + self.genuses, self.species) 
        
        
    def forward(self, x): 
        x = self.pool2(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool5(F.relu(self.conv4(x)))
        #x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        #TODO: relu help or not?
        fam = F.relu(self.famfc(x))
        gen = F.relu(self.genfc(torch.cat([fam, x] ,1)))
        spec = self.specfc(torch.cat([gen, x], 1))
        return(spec, gen, fam)
    

class SkipFCNet(nn.Module):
    """
    Checking - it requires more training time, 1 layer more 
    """
    def __init__(self, species, families, genuses, num_channels):
    #TODO: add 2 fc layers before fgs downsampling
        super(SkipFCNet, self).__init__()
        self.categories=species
        self.species = species
        self.families = families
        self.genuses = genuses
        self.num_channels=num_channels
        self.conv1 = nn.Conv2d(self.num_channels, 64, 7,1,1) # try a kernel of size 7 like TNN model
        self.conv2 = nn.Conv2d(64, 128, 3,1,1)
        self.conv3 = nn.Conv2d(128, 256, 3,1,1)
        self.conv4 = nn.Conv2d(256, 256, 3,1,1)        
        self.conv5 = nn.Conv2d(256, 512, 3,1,1)        
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool5 = nn.MaxPool2d(5, 5)
        self.choke1 = 120
        self.choke2 = 84
        self.fc1 = nn.Linear(256 * 6 * 6, self.choke1)
        self.fc2 = nn.Linear(120, self.choke2) 
        self.famfc = nn.Linear(self.choke2, self.families) 
        self.genfc = nn.Linear(self.choke2+self.families, self.genuses)
        self.specfc = nn.Linear(self.choke2+self.genuses, self.species) 
        
        
    def forward(self, x): 
        x = self.pool2(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool5(x)
        #x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        fam = F.relu(self.famfc(x))
        gen = F.relu(self.genfc(torch.cat([fam, x] ,1)))
        spec = self.specfc(torch.cat([gen, x], 1))
        return(spec, gen, fam)


class MixNet(nn.Module):
    """
    Checking - it requires more training time, 1 layer more 
    """
    def __init__(self, species, families, genuses, num_channels, env_rasters):
    #inspo: https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/ 
        super(MixNet, self).__init__()
        self.categories=species
        self.species = species
        self.families = families
        self.genuses = genuses
        self.num_channels=num_channels
        self.conv1 = nn.Conv2d(self.num_channels, 64, 7,1,1) # try a kernel of size 7 like TNN model
        self.conv2 = nn.Conv2d(64, 128, 3,1,1)
        self.conv3 = nn.Conv2d(128, 256, 3,1,1)
        self.conv4 = nn.Conv2d(256, 256, 3,1,1)        
        self.conv5 = nn.Conv2d(256, 512, 3,1,1)        
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool5 = nn.MaxPool2d(5, 5)
        self.cnn_choke1 = 256
        self.cnn_choke2 = 128
        self.mlp_choke1 = 64
        self.mlp_choke2 = 128
        self.bottleneck = self.mlp_choke2 + self.cnn_choke2
        self.mlp1 = nn.Linear(env_rasters, self.mlp_choke1)
        self.mlp2(self.mlp_choke1, self.mlp_choke2)
        self.fc1 = nn.Linear(256 * 5 * 5, self.cnn_choke1)
        self.fc2 = nn.Linear(self.cnn_choke1, self.cnn_choke2) 
        self.famfc = nn.Linear(self.bottleneck, self.families) 
        # does this add the values together or 
        self.genfc = nn.Linear(self.bottleneck+self.families, self.genuses)
        self.specfc = nn.Linear(self.bottleneck+self.genuses, self.species) 
        
        
    def forward(self, img, rasters):
        # pass images through CNN
        x = self.pool2(F.relu(self.conv1(imgs)))
        x = self.pool2(F.relu(self.conv2(x)))
        #x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool5(x)
        #x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # pass raster data through MLP
        y = F.relu(self.mlp1(rasters))
        y = F.relu(self.mlp2(y))
        combined = torch.cat([x,y],1)
        fam = F.relu(self.famfc(combined))
        gen = F.relu(self.genfc(torch.cat([fam, combined] ,1)))
        spec = self.specfc(torch.cat([gen, combined], 1))
        return(spec, gen, fam)    
    
    
    
class OGNoFamNet(nn.Module):
    """
    Checking - it requires more training time, 1 layer more 
    """
    def __init__(self, species, genuses, num_channels):

        super(OGNoFamNet, self).__init__()
        self.categories=species
        self.species = species
        self.genuses = genuses
        self.num_channels=num_channels
        self.conv1 = nn.Conv2d(self.num_channels, 64, 7,1,1) # try a kernel of size 7 like TNN model
        self.conv2 = nn.Conv2d(64, 128, 3,1,1)
        self.conv3 = nn.Conv2d(128, 256, 3,1,1)
        self.conv4 = nn.Conv2d(256, 256, 3,1,1)        
        self.conv5 = nn.Conv2d(256, 512, 3,1,1)        
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool5 = nn.MaxPool2d(5, 5)
        self.genfc = nn.Linear(256*6*6, self.genuses) 
#         self.genfc = nn.Linear(self.families, self.genuses)
        self.specfc = nn.Linear(self.genuses, self.species) 
        
        
    def forward(self, x): 
        x = self.pool2(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool5(x)
        #x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        gen = F.relu(self.genfc(x))
#         gen = F.relu(self.genfc(fam))
        spec = self.specfc(gen)
        return(spec, gen)
