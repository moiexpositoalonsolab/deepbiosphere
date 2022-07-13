import os
import numpy as np
import operator
from functools import reduce
import deepbiosphere.VGG as vgg
import deepbiosphere.ResNet as resnet
import deepbiosphere.TResNet as tresnet
import deepbiosphere.Inception as inception
import deepbiosphere.Utils as utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

    # -------- CNN Models ------------- #

# reimplement standard VGGNet with Hsu initialization here
def ResNet_18(pretrained, arch_type, species, genera, families, base_dir):
    # convert pretrained to bool
    # initialize right network with arch_type and num_channels
    # check if feature extraction or finetuning
    return resnet.resnet18(pretrained, arch_type, species, genera, families, base_dir)


def ResNet_34(pretrained, arch_type, species, genera, families, base_dir):
    # convert pretrained to bool
    # initialize right network with arch_type and num_channels
    # check if feature extraction or finetuning
    return resnet.resnet34(pretrained, arch_type, species, genera, families, base_dir)

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

def VGG_16(pretrained, batch_norm, species, families, genera, arch_type, base_dir):
    # convert pretrained to bool
    # convert batch_norm to bool
    # initialize right network with arch_type and num_channels
    # check if feature extraction or finetuning
    if batch_norm:
        return vgg.vgg16(species, genera, families, base_dir, arch_type, pretrained)
    else:
        return vgg.vgg16_bn(species, genera, families, base_dir, arch_type, pretrained)

def TResNet_M(pretrained, num_spec, num_gen, num_fam, base_dir):
    return tresnet.TResnetM(pretrained, num_spec, num_gen, num_fam, base_dir)

def TResNet_L(pretrained, num_spec, num_gen, num_fam, base_dir):
    return tresnet.TResnetL(pretrained, num_spec, num_gen, num_fam, base_dir)

    # -------- Baselines ------------- #

# species only inception, Botella baseline
def InceptionV3(pretrained, num_spec, num_gen, num_fam, base_dir):
    return inception.Inception3(num_classes=num_spec)
 
    
class Bioclim_MLP(nn.Module):
    """
    Fully-connected Multilayered perceptron trained from solely bioclim
    """
    def __init__(self, null, num_spec, num_gen, num_fam, env_rasters=19, nlayers=4, drop=.25, base_dir=None):

        super(Bioclim_MLP, self).__init__()
        self.families = num_fam
        self.genera = num_gen
        self.species = num_spec
        self.env_rasters = env_rasters
        self.mlp_choke1 = 1000
        self.mlp_choke2 = 2000
        self.elu = nn.ELU()
        layers = []
        layers.append(nn.Linear(env_rasters, self.mlp_choke1))
        layers.append(self.elu)
        # smaller layers
        for i in range(1, (nlayers//2)):
            layers.append(nn.Linear(self.mlp_choke1, self.mlp_choke1))
            layers.append(self.elu)
        # setup to bigger layers
        layers.append(nn.Linear(self.mlp_choke1, self.mlp_choke2))
        layers.append(self.elu)
        # and dropout
        layers.append(nn.Dropout(drop))
        # bigger layers
        for i in range((nlayers//2)+1, nlayers):
            layers.append(nn.Linear(self.mlp_choke2, self.mlp_choke2))
            layers.append(self.elu)

        self.layers = nn.Sequential(*layers)
        self.mlp_fam = nn.Linear(self.mlp_choke2, self.families)
        self.mlp_gen = nn.Linear(self.mlp_choke2, self.genera)
        self.mlp_spec = nn.Linear(self.mlp_choke2, self.species)

    def forward(self, rasters):
        # pass images through CNN
        x = self.layers(rasters)
        fam = self.mlp_fam(x)
        gen = self.mlp_gen(x)
        spec = self.mlp_spec(x)
        return spec, gen, fam

    # -------- Joint Models ------------- #

def Joint_TResNet_M(pretrained, num_spec, num_gen, num_fam, env_rasters, base_dir):
    return tresnet.Joint_TResNetM(pretrained, num_spec, num_gen, num_fam, env_rasters, base_dir)

def Joint_TResNet_L(pretrained, num_spec, num_gen, num_fam, env_rasters, base_dir):
    return tresnet.Joint_TResNetL(pretrained, num_spec, num_gen, num_fam, env_rasters, base_dir)

def Joint_ResNet_18(pretrained, num_spec, num_gen, num_fam, env_rasters):
    return resnet.joint_resnet18(pretrained, num_spec, num_gen, num_fam, env_rasters)

def Joint_VGG11_MLP(species, genera, families, base_dir, batch_norm, arch_type, pretrained, env_rasters):
    return vgg.joint_vgg11(species, genera, families, base_dir, batch_norm, arch_type, env_rasters, pretrained)

def Joint_VGG16_MLP(species, genera, families, base_dir, batch_norm, arch_type, pretrained, env_rasters):
    return vgg.joint_vgg16(species, genera, families, base_dir, batch_norm, arch_type, env_rasters, pretrained)


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
    base_dir = utils.setup_pretrained_dirs(base_dir)
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
