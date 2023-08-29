# deepbiosphere functions
import deepbiosphere.Utils as utils
import deepbiosphere.TResNet as tresnet
import deepbiosphere.Bioclim_MLP as bioclim_mlp
from deepbiosphere.Utils import paths
import deepbiosphere.Inception as inception

#  torch  stats functions
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# misc functions
import os
import operator
from functools import reduce, partial



# -------- CNN Models ------------- #

# Hacky, but this wayI can have a separate file for each CNN model and just
# have a simple function to instantiate within the same "models" file

def Deepbiosphere(num_spec, num_gen, num_fam, env_rasters, pretrained=None, base_dir=None): 
    return tresnet.Joint_TResNetM(num_spec, num_gen, num_fam, env_rasters, pretrained, base_dir)

def RS_Only_TResNet(num_spec, num_gen, num_fam, pretrained=tresnet.Pretrained.NONE, base_dir=None): 
    return tresnet.TResnetM(num_spec, num_gen, num_fam, pretrained, base_dir)

# -------- Baselines ------------- #

# species only inception, Botella baseline
def InceptionV3(num_spec):
    return inception.Inception3(num_classes=num_spec)
 
def Bioclim_MLP(num_spec, num_gen, num_fam, env_rasters):
    return bioclim_mlp(num_spec, num_gen, num_fam, env_rasters)

# ---------- Types ---------- #

# models that can be run
class Model(utils.FuncEnum, metaclass=utils.MetaEnum):
    RS_TRESNET =  partial(RS_Only_TResNet)
    DEEPBIOSPHERE = partial(Deepbiosphere)
    BIOCLIM_MLP = partial(Bioclim_MLP)
    INCEPTION = partial(InceptionV3)

# ------------ Helper Methods ------------- #
    
def get_pretrained_models(base_dir):
    # TODO: auto download all the pretrained models here
    raise NotImplemented
