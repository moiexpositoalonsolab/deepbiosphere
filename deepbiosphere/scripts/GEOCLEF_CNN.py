"""
Defines Deep Neural Network implementations with torch for
Biodiversity Geo-Modeling

@author: moisesexpositoalonso@gmail.com
"""


import os
import numpy as np
import operator
from functools import reduce
import deepbiosphere.scripts.GEOCLEF_Config as config
import deepbiosphere.scripts.VGG as vgg
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


# --------- CNN Models ------------ # 

class SpecOnly(nn.Module):
    """
    Checking - it requires more training time, 1 layer more 
    """
    def __init__(self, species, num_channels):

        super(SpecOnly, self).__init__()
        self.categories=species
        self.species = species
        self.num_channels=num_channels
        self.conv1 = nn.Conv2d(self.num_channels, 64, 7,1,1) # try a kernel of size 7 like TNN model
        self.conv2 = nn.Conv2d(64, 128, 3,1,1)
        self.conv3 = nn.Conv2d(128, 256, 3,1,1)
        self.conv4 = nn.Conv2d(256, 256, 3,1,1)        
        self.conv5 = nn.Conv2d(256, 512, 3,1,1)        
        self.pool2 = nn.MaxPool2d(2, 2)
                               
        self.pool5 = nn.MaxPool2d(5, 5)
        self.chokepoint = 256*6*6
        self.specfc = nn.Linear(self.chokepoint, self.species) 
        
        
    def forward(self, x): 
        x = self.pool2(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool5(F.relu(self.conv4(x)))
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        spec = self.specfc(x)
        return(spec)    
    
    
# create supermodel from torchvision    
#     'ResNet', 'VGG_11',  'VGG_16'],
# TODO: use torchvision to get pretrained VGGNet, ResNet
# also TODO: get un-trained ResNet
# problem: have to figure out how to load paramters for pretrained in properly then add s,g,f layers

# reimplement standard VGGNet with Hsu initialization here
def ResNet(pretrained, species, families, genera, num_channels, arch_type):
    # convert pretrained to bool
    # initialize right network with arch_type and num_channels
    # check if feature extraction or finetuning
    raise NotImplementedError
    
def VGG_11(pretrained, batch_norm, species, families, genera, arch_type, base_dir):
    # convert pretrained to bool
    # convert batch_norm to bool
    # deal with extra band for pretrained
    # initialize right network with arch_type and num_channels
    # check if feature extraction or finetuning
    if batch_norm:
        return vgg.vgg11(species, genera, families, base_dir, arch_type, pretrained)
    else:
        return vgg.vgg11_bn(species, genera, families, base_dir, arch_type, pretrained)

def VGG_16(pretrained, batch_norm, species, families, genera, arch_type):
    # convert pretrained to bool
    # convert batch_norm to bool
    # initialize right network with arch_type and num_channels
    # check if feature extraction or finetuning
    if batch_norm:
        return vgg.vgg16(species, genera, families, base_dir, arch_type, pretrained)
    else:
        return vgg.vgg16_bn(species, genera, families, base_dir, arch_type, pretrained)


    
    # TODO: fix to have normal convblock nature, well this is kind of a copy of VGGNet architecture in a way?
    # may just move forward with VGGNet  from now on
class FlatNet(nn.Module):
    """
    Checking - it requires more training time, 1 layer more 
    """
    def __init__(self, species, families, genera, num_channels):

        super(FlatNet, self).__init__()
        self.categories=species
        self.species = species
        self.families = families
        self.genera = genera
        self.num_channels=num_channels
      
        # in channels, out channels, kernel size, stride, padding, dilation
        self.conv1 = nn.Conv2d(self.num_channels, 64, 7,1,1) # try a kernel of size 7 like TNN model
        self.conv2 = nn.Conv2d(64, 128, 3,1,1)
        self.conv3 = nn.Conv2d(128, 256, 3,1,1)
        self.conv4 = nn.Conv2d(256, 256, 3,1,1)        
        self.conv5 = nn.Conv2d(256, 512, 3,1,1)        
        self.pool2 = nn.MaxPool2d(2, 2)
                               
        self.pool5 = nn.MaxPool2d(5, 5)
        self.chokepoint = 256*6*6
        self.famfc = nn.Linear(self.chokepoint, self.families) 
        self.genfc = nn.Linear(self.chokepoint, self.genera)
        self.specfc = nn.Linear(self.chokepoint, self.species) 
        
    # typical conv block is conv relu conv relu maxpool
    # for resnet: conv, batchnorm, relu, etc.
    def forward(self, x): 
        x = self.pool2(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        #x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = self.pool5(F.relu(self.conv4(x)))
        #x = self.pool5(F.relu(self.conv5(x)))
        # TODO: see how big final layer of VGGNet is
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        #TODO: relu help or not?
        fam = self.famfc(x)
        gen = self.genfc(x)
        spec = self.specfc(x)
        return(spec, gen, fam)
    
    
    
    # ------ Baselines --------- # 
class MLP_Family(nn.Module):
    """
    Checking - it requires more training time, 1 layer more 
    """
    def __init__(self, families, env_rasters):
    #inspo: https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/ 
        super(MLP_Family, self).__init__()
        self.families = families
        self.env_rasters = env_rasters
        self.mlp_choke1 = 64
        self.mlp_choke2 = 128
        self.mlp1 = nn.Linear(env_rasters, self.mlp_choke1)
        self.mlp2 = nn.Linear(self.mlp_choke1, self.mlp_choke2)
        self.mlpout = nn.Linear(self.mlp_choke2, self.families)        
        
    def forward(self, rasters):
        # pass images through CNN
        x = F.relu(self.mlp1(rasters))
        x = F.relu(self.mlp2(x))
        fam = self.mlpout(x)
        return fam
    
    
class MLP_Family_Genus(nn.Module):
    """
    Checking - it requires more training time, 1 layer more 
    """
    def __init__(self, families, genera, env_rasters):
    #inspo: https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/ 
        super(MLP_Family_Genus, self).__init__()
        self.families = families
        self.genera = genera
        self.env_rasters = env_rasters
        self.mlp_choke1 = 64
        self.mlp_choke2 = 128
        self.mlp1 = nn.Linear(env_rasters, self.mlp_choke1)
        self.mlp2 = nn.Linear(self.mlp_choke1, self.mlp_choke2)
        self.mlp_fam = nn.Linear(self.mlp_choke2, self.families)
        self.mlpout = nn.Linear(self.families, self.genera)
        
    def forward(self, rasters):
        # pass images through CNN
        x = F.relu(self.mlp1(rasters))
        x = F.relu(self.mlp2(x))
        fam = self.mlp_fam(F.relu(x))
        gen = self.mlpout(F.relu(fam))        
        return fam, gen
    
       
class MLP_Family_Genus_Species(nn.Module):
    """
    Checking - it requires more training time, 1 layer more 
    """
    def __init__(self, families, genera, species, env_rasters):
    #inspo: https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/ 
        super(MLP_Family_Genus_Species, self).__init__()
        self.families = families
        self.genera = genera
        self.species = species
        self.env_rasters = env_rasters
        self.mlp_choke1 = 64
        self.mlp_choke2 = 128
        self.mlp1 = nn.Linear(env_rasters, self.mlp_choke1)
        self.mlp2 = nn.Linear(self.mlp_choke1, self.mlp_choke2)
        self.mlp_fam = nn.Linear(self.mlp_choke2, self.families)
        self.mlp_gen = nn.Linear(self.families, self.genera)
        self.mlpout = nn.Linear(self.genera, self.species)
        
    def forward(self, rasters):
        # pass images through CNN
        x = F.relu(self.mlp1(rasters))
        x = F.relu(self.mlp2(x))
        fam = self.mlp_fam(F.relu(x))
        gen = self.mlp_gen(F.relu(fam))
        spec = self.mlpout(F.relu(gen))        
        return spec, gen, fam

    
    # -------- Joint Models ------------- #
    
    # TODO: modify this into a jointly trained model
class MixNet(nn.Module):
    """
    Checking - it requires more training time, 1 layer more 
    """
    def __init__(self, species, families, genera, num_channels, env_rasters):
    #inspo: https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/ 
        super(MixNet, self).__init__()
        self.categories=species
        self.species = species
        self.families = families
        self.genera = genera
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
        self.mlp2 = nn.Linear(self.mlp_choke1, self.mlp_choke2)
        self.fc1 = nn.Linear(256 * 6 * 6, self.cnn_choke1)
        self.fc2 = nn.Linear(self.cnn_choke1, self.cnn_choke2) 
        self.famfc = nn.Linear(self.bottleneck, self.families) 
        # does this add the values together or 
        self.genfc = nn.Linear(self.bottleneck+self.families, self.genera)
        self.specfc = nn.Linear(self.bottleneck+self.genera + self.families, self.species) 
        
        
    def forward(self, img, rasters):
        # pass images through CNN
        x = self.pool2(F.relu(self.conv1(img)))
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
        gen = F.relu(self.genfc(torch.cat([combined, fam] ,1)))
        spec = self.specfc(torch.cat([combined, gen, fam], 1))
        return(spec, gen, fam)    
    
class MixFullNet(nn.Module):
    """
    Checking - it requires more training time, 1 layer more 
    """
    
    def __init__(self, species, families, genera, num_channels, env_rasters):
    #inspo: https://www.pyimagesearch.com/2019/02/04/keras-multiple-inputs-and-mixed-data/ 
        super(MixFullNet, self).__init__()
        self.categories=species
        self.species = species
        self.families = families
        self.genera = genera
        self.num_channels=num_channels
        self.conv1 = nn.Conv2d(self.num_channels, 64, 7,1,1) # try a kernel of size 7 like TNN model
        self.conv2 = nn.Conv2d(64, 128, 3,1,1)
        self.conv3 = nn.Conv2d(128, 256, 3,1,1)
        self.conv4 = nn.Conv2d(256, 256, 3,1,1)        
        self.conv5 = nn.Conv2d(256, 512, 3,1,1)        
        self.pool2 = nn.MaxPool2d(2, 2)
        self.pool5 = nn.MaxPool2d(5, 5)
        
        self.chokepoint = 256*6*6


        self.mlp_choke1 = 48
        self.mlp_choke2 = 96

        self.bottleneck = self.mlp_choke2 + self.chokepoint
        self.mlp1 = nn.Linear(env_rasters, self.mlp_choke1)
        self.mlp2 = nn.Linear(self.mlp_choke1, self.mlp_choke2)
        self.famfc = nn.Linear(self.bottleneck, self.families) 
        # does this add the values together or 
        self.genfc = nn.Linear(self.bottleneck+self.families, self.genera)
        self.specfc = nn.Linear(self.bottleneck+self.genera + self.families, self.species) 
        
        
    def forward(self, img, rasters):
        # pass images through CNN
        x = self.pool2(F.relu(self.conv1(img)))
        x = self.pool2(F.relu(self.conv2(x)))
        #x = F.relu(self.conv2(x))
        x = self.pool2(F.relu(self.conv3(x)))
        x = F.relu(self.conv4(x))
        x = self.pool5(x)
        #x = self.pool5(F.relu(self.conv5(x)))
        x = x.view(x.shape[0], x.shape[1]*x.shape[2]*x.shape[3])
        # pass raster data through MLP
        y = F.relu(self.mlp1(rasters))
        y = F.relu(self.mlp2(y))
        combined = torch.cat([x,y],1)
        fam = F.relu(self.famfc(combined))
        gen = F.relu(self.genfc(torch.cat([combined, fam] ,1)))
        spec = self.specfc(torch.cat([combined, gen, fam], 1))
        return(spec, gen, fam)    
    

    
    
    # ------------ Helper Methods ------------- # 
def get_pretrained_models(base_dir):
    
    # this code works with a version of torch that's compatible with python 3.5, 
    # specifically torch==1.2.0 and tv==0.4.0
    # the versions that are on calc are torch==1.4.0 and tv==0.5.0
    # TODO: see if this works on calc too

    # url grabbed from below on 1/8/2021
    # https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
    pret_resnet = "https://download.pytorch.org/models/resnet18-5c106cde.pth"
    # urls grabbed from below on 1/8/2021
    # https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
    pret_vggnet = "https://download.pytorch.org/models/vgg11-bbd30ac9.pth"
    pret_vggbn = "https://download.pytorch.org/models/vgg11_bn-6002323d.pth"
    base_dir = config.setup_pretrained_dirs(base_dir)
    res_dir = base_dir + 'ResNet/'
    vgg_dir = base_dir + 'VGGNet/'
    if torch.__version__=='1.2.0':
        torch.hub.set_dir(base_dir)
        torch.utils.model_zoo.load_url(pret_resnet, model_dir=res_dir)
        torch.utils.model_zoo.load_url(pret_vggnet, model_dir=vgg_dir)
        torch.utils.model_zoo.load_url(pret_vggbn, model_dir=vgg_dir)
    elif torch.__version__ == '1.4.0':
        torch.hub.set_dir(base_dir)
        torch.hub.load_state_dict_from_url(pret_resnet, model_dir=res_dir)
        torch.hub.load_state_dict_from_url(pret_vggnet, model_dir=vgg_dir)
        torch.hub.load_state_dict_from_url(pret_vggbn, model_dir=vgg_dir) 
    else:
        raise NotImplementedError("your version of pytorch is incompatible, must be either version 1.2.0 or 1.4.0. Either change to a supported version of torch or modify the code for your own version")
