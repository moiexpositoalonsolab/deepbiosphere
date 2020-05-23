from __future__ import print_function, division
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def get_gbif_data(pth, split, country):
    ## Grab GBIF observation data
    obs_pth = f"{pth}occurrences/occurrences_{country}_{split}.csv"
    return pd.read_csv(obs_pth, sep=';')  

def us_image_from_id(id_, pth):
    abcd = id_ % 10000
    ab, cd = math.floor(abcd/100), abcd%100
    cdd = math.ceil((cd+ 1)/5)
    cdd = f"0{cdd}"  if cdd < 10 else f"{cdd}"
    ab = f"0{ab}" if id_ / 1000 > 1 and ab < 10 else ab
    cd = f"0{cd}" if id_ / 1000 > 1 and cd < 10 else cd
    subpath = f"patches_us_{cdd}/{cd}/{ab}/"
    alt = f"{pth}{subpath}{id_}_alti.npy"
    rgbd = f"{pth}{subpath}{id_}.npy"    
    np_al = np.load(alt)
    np_img = np.load(rgbd)
    np_al = np.expand_dims(np_al, 2)
    np_all = np.concatenate((np_al, np_img), axis=2)
    return np_all

def fr_img_from_id(id_, pth):
    raise NotImplementedError

def us_image_from_id(id_, pth):
    abcd = id_ % 10000
    ab, cd = math.floor(abcd/100), abcd%100
    cdd = math.ceil((cd+ 1)/5)
    cdd = f"0{cdd}"  if cdd < 10 else f"{cdd}"
    ab = f"0{ab}" if id_ / 1000 > 1 and ab < 10 else ab
    cd = f"0{cd}" if id_ / 1000 > 1 and cd < 10 else cd
    subpath = f"patches_us_{cdd}/{cd}/{ab}/"
    alt = f"{pth}{subpath}{id_}_alti.npy"
    rgbd = f"{pth}{subpath}{id_}.npy"    
    np_al = np.load(alt)
    np_img = np.load(rgbd)
    np_al = np.expand_dims(np_al, 2)
    np_all = np.concatenate((np_al, np_img), axis=2)
    return np_all

def prep_US_data(us_obs):
    spec_2_id = {k:v for k, v in zip(us_obs.species_id.unique(), np.arange(len(us_obs.species_id.unique())))}
    us_obs['species_id'] = us_obs['species_id'].map(spec_2_id)

class GeoClefDataset(Dataset):
    def __init__(self, obs_path, split='train', country='us', img_dir, transform=None):
        
        obs = get_gbif_data(obs_path, split, country)
        if country == 'us':
            obs = prep_US_data(obs)
        # Grab only obs id, species id because lat /lon not necessary at the moment
        self.obs = us_obs[['id', 'species_id']].to_numpy()
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        id_, label = self.obs[idx, 0], self.obs[idx, 1]
        images = us_image_from_id(id_, self.img_dir) 
        if self.transform:
            images = self.transform(images)
        return (label, images)



