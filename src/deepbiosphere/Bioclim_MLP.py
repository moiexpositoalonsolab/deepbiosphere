#  torch  stats functions
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class Bioclim_MLP(nn.Module):
    """
    Fully-connected Multilayered perceptron trained from solely bioclim
    Architecture inspired by Battey et al. 2019 Predicting Geographic Location 
    From Genetic Variation with Deep Neural Networks
    """
    def __init__(self, num_spec, num_gen, num_fam, env_rasters=19, nlayers=4, drop=.25):

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
        if self.species != -1:
            self.mlp_spec = nn.Linear(self.mlp_choke2, self.species)
        if self.genera != -1:
            self.mlp_gen = nn.Linear(self.mlp_choke2, self.genera)
        if self.families != -1:
            self.mlp_fam = nn.Linear(self.mlp_choke2, self.families)
        
        

    def forward(self, rasters):
        x = self.layers(rasters)
        spec = self.mlp_spec(x)

        if not self.training:
            return spec
        # use all 3 taxonomic levels for training
        if (self.genera != -1) & (self.families != -1):
            fam = self.mlp_fam(x)
            gen = self.mlp_gen(x)
            return (spec, gen, fam)
        # use only species genus taxonomic levels
        elif (self.families != -1):
            gen = self.mlp_gen(x)
            return (spec, gen)
        # use only species taxonomic level
        else:
            return ([spec])
