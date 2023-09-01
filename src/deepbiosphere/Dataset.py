# GIS packages
import rasterio
import geopandas as gpd
from pyproj import Proj
from scipy.spatial import cKDTree
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, box

# deepbiosphere packages
import deepbiosphere.NAIP_Utils  as naip
import deepbiosphere.Build_Data as build
from deepbiosphere.Utils import paths
import deepbiosphere.Utils as utils

# torch / stats packages
import torch
import numpy as np
import pandas as pd
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset as TorchDataset

# miscellaneous packages
import os
import time
import math
import glob
import json
import copy
import random
from enum import Enum
from tqdm import tqdm
from functools import partial
from types import SimpleNamespace

## ---------- MAGIC NUMBERS ---------- ##
# image size when fivecropping
FC_SIZE = 128
# possible rotation options for transform augmentations
DEGS = [15,30,45, 60,-75, 90]
# the kernel to use for gaussian blurring and sharpening
GKS = 5


# ---------- Static Types ---------- #
    
# choices=['bioclim', 'naip', 'joint_naip_bioclim']
class DataType(Enum, metaclass=utils.MetaEnum):
    BIOCLIM = 'bioclim'
    NAIP = 'naip'
    JOINT_NAIP_BIOCLIM = 'joint_naip_bioclim'

# choices=['multi_species', 'single_species', 'single_label']
class DatasetType(Enum, metaclass=utils.MetaEnum):
    MULTI_SPECIES = 'multi_species'
    SINGLE_SPECIES = 'single_species'
    SINGLE_LABEL = 'single_label'
    
    
# ---------- Data augmentations ---------- #

def random_augment(self,img):
    aug_choice = random.sample(range(4),1)
    # first choice: tencrop augmentation
    if aug_choice == 0:
        vert = random.random() > 0.5
        options = T.tencrop(size=(FC_SIZE,FC_SIZE), vertical_flip=vert) (torch.tensor(img))
        choice = random.sample(range(len(options)), 1)
        img = options[choice[0]]
        # now resample back up to 256x256
        img = TF.resize(img, (utils.IMG_SIZE,utils.IMG_SIZE))
    # second choice: rotation augmentation
    if aug_choice == 1:
        choice = random.sample((len(DEGS)), 1)
        img = TF.rotate(torch.tensor(batches[i,:,:,:]), DEGS[choice[0]])
        # half the time, also flip
        if random.random() < 0.5:
            # half the time flip vertical, other half horizontal
            img = TF.vflip(img) if random.random() < 0.5 else TF.hflip(img)
    # third choice: sharpen / blur augmentation
    if aug_choice == 2:
        # 50/50 chance blur vs. sharpen
        if random.random() > 0.5:
            img = TF.gaussian_blur(torch.tensor(img), kernel_size=GKS)
        else:
            # sharpening can only handle 1,3 channels so have to stitch together to get 4 channels
            img = torch.vstack((TF.adjust_sharpness(torch.tensor(img[:3,:,:]),GKS), TF.adjust_sharpness(torch.tensor(batches[3:4,:,:]),GKS)))
    # fourth choice: do nothing!
    else:
        img = torch.tensor(img)
    return img


def fivecrop_augment(img):
    # crop 5x
    imgs  = TF.five_crop(torch.tensor(img), size=(FC_SIZE,FC_SIZE))
    # pick a random crop to return
    which = random.sample(range(len(imgs)), 1)
    return imgs[which[0]]


# ---------- Function Types ---------- #

# choices=['none', 'random', 'fivecrop'])
# TODO: convert to function type and apply transform that way
# then, TODO convert type checks here and in run / inference / map making
# To use these types...
# valid augments
class Augment(utils.FuncEnum, metaclass=utils.MetaEnum):
    
    NONE = partial(utils.pass_)
    RANDOM = partial(random_augment)
    FIVECROP = partial(fivecrop_augment)    

# gross but pandas stores lists in csvs
# as strings, so need to unread that string
def parse_string_to_tuple(string):
    return eval(string)


# ---------- Data helper methods ---------- #

def map_index(index):
     return {
            k:v for k, v in
            zip(np.arange(len(index)), index)
        }

def parse_string_to_string(string):
    string = string.replace("{", '').replace("}", "").replace("'", '').replace("[", '').replace("]", '').replace(")", '').replace("(", '')
    split = string.split(", ")
    return split

def parse_string_to_int(string):
    string = string.replace("{", '').replace("}", "").replace("'", '').replace("[", '').replace("]", '').replace(")", '').replace("(", '')
    split = string.split(", ")
    return [int(s) for s in split]

def parse_string_to_float(string):
    string = string.replace("{", '').replace("}", "").replace("'", '').replace("[", '').replace("]", '').replace(")", '').replace("(", '')
    split = string.split(", ")
    return [float(s) for s in split]

def load_metadata(dataset_name, parent_dir=None):
    if parent_dir is None:
        path = f"{paths.OCCS}{dataset_name}_metadata.json"
    else:
        path = f"{parent_dir}{dataset_name}_metadata.json"
    return SimpleNamespace(**json.load(open(path, 'r')))
 
def get_onehot(species_id, genus_id, family_id, nspec, ngen, nfam):
    all_specs, all_gens, all_fams = [], [], []
    for spids, gids, fids in zip(species_id, genus_id, family_id):
        specs_tens = np.full((nspec), 0)
        specs_tens[spids] += 1
        all_specs.append(specs_tens)
        gens_tens = np.full((ngen), 0)
        gens_tens[gids] += 1
        all_gens.append(gens_tens)
        fams_tens = np.full((nfam), 0)
        fams_tens[fids] += 1
        all_fams.append(fams_tens)
    all_specs = torch.tensor(np.stack(all_specs))
    all_gens = torch.tensor(np.stack(all_gens))
    all_fams = torch.tensor(np.stack(all_fams))
    return all_specs, all_gens, all_fams

    
def get_specnames(metadata):
    # sort 0-N
    sortd = sorted(metadata.spec_2_id.items(), key = lambda x: x[1])
    # and grab species names
    return list(zip(*sortd))[0]

def check_bioclim(daset, metadata, state):
    # for when using bioclim data, read in the bioclim rasters
    # usually the dataset will have the values pre-computed
    bio_cols = [c for c in daset.columns if '_bio' in c]
    # TODO: ensure variables are in the same order each time?
    if len(bio_cols) > 1:
        return torch.tensor(daset[bio_cols].values) # self.bioclim
    else:
        return load_bioclim(daset, metadata, state)
        
        
def load_bioclim(daset, metadata, state): 
    
    rasters = build.get_bioclim_rasters(state=state)
    pts = [Point(lon, lat) for lon, lat in zip(daset[metadata.loName], daset[metadata.latName])]
    # GBIF returns coordinates in WGS84 according to the API
    # https://www.gbif.org/article/5i3CQEZ6DuWiycgMaaakCo/gbif-infrastructure-data-processing
    daset = gpd.GeoDataFrame(daset, geometry=pts, crs=naip.CRS.GBIF_CRS)
    #  precompute  the bioclim variables at each test locatoin
    bioclim = []
    for point in tqdm(daset.geometry,total=len(daset), unit='point'):
        curr_bio = []
        # since we've confirmed all the rasters have identical
        # transforms, can just calculate the x,y coord once
        x,y = rasterio.transform.rowcol(rasters[0][1], *point.xy)
        for j, (ras, transf,_,_) in enumerate(rasters):
            curr_bio.append(ras[0,x,y])
        bioclim.append(curr_bio)
    return torch.tensor(np.squeeze(np.stack(bioclim)))

# ---------- Actual dataset class ---------- #

class DeepbioDataset(TorchDataset):
    
    def __init__(self, dataset_name, datatype, dataset_type, state, year, band, split, augment, prep_onehots=True):

        # load in observations & metadata
        daset = pd.read_csv(f"{paths.OCCS}{dataset_name}.csv")
        metadata = load_metadata(dataset_name)
        # pandas saves lists as strings in csv, gotta parse back to strings
        parsed = [parse_string_to_int(s) for s in daset.specs_overlap_id]
        daset['specs_overlap_id'] = parsed
        parsed = [parse_string_to_int(s) for s in daset.gens_overlap_id]
        daset['gens_overlap_id'] = parsed
        parsed = [parse_string_to_int(s) for s in daset.fams_overlap_id]
        daset['fams_overlap_id'] = parsed
        # every species in dataset
        # is guaranteed to be in the species
        # column so this is chill for now
        self.band = band
        self.name = dataset_name
        self.nspec = len(metadata.spec_2_id)
        self.ngen = len(metadata.gen_2_id)
        self.nfam =  len(metadata.fam_2_id)
        self.metadata = metadata
        self.datatype = DataType[datatype]
        self.dataset_type = DatasetType[dataset_type]
        self.total_len = len(daset)
        # for when using remote sensing data, read in NAIP statistics
        # TODO: change depending on your pretraining
        self.mean = metadata.dataset_means[f"naip_{year}"]['means']
        self.std = metadata.dataset_means[f"naip_{year}"]['stds']
        self.ids = daset[metadata.idCol]
        # only relevant for cases where remote sensing data used
        self.augment = Augment[augment]
        # split data 
        # if band is >=0, means to use the banding split
        # if band = -1, then use the uniform spatial split
        if split != 'all_points':
            daset =  daset[daset[f"{split}_{band}"]] if band >=0 else daset[daset.unif_train_test == split]
        self.len_dset = len(daset)
        self.ids = daset[metadata.idCol]
        self.pres_specs = set([spec for sublist in daset.specs_overlap_id for spec in sublist])
        print(f"{split} dataset has {self.len_dset} points")
        # next, map the indices and save them as numpy
        self.idx_map = map_index(daset.index)
        self.index = daset.index
        if prep_onehots:
            # get onehot image labels
            # WARNING! Super memory inefficient, will need to fix
            all_specs, all_gens, all_fams = get_onehot(daset.species_id, daset.genus_id, daset.family_id, self.nspec, self.ngen, self.nfam)
            self.all_specs_single = all_specs
            self.all_gens_single = all_gens
            self.all_fams_single = all_fams
            # get multihot image labels
            all_specs, all_gens, all_fams = get_onehot(daset.specs_overlap_id, daset.gens_overlap_id, daset.fams_overlap_id, self.nspec, self.ngen, self.nfam)
            self.all_specs_multi = all_specs
            self.all_gens_multi = all_gens
            self.all_fams_multi = all_fams
            # finally, we'll precompute the species labels for topk testing
            self.specs = torch.tensor(daset.species_id.tolist())
            self.gens = torch.tensor(daset.genus_id.tolist())
            self.fams = torch.tensor(daset.family_id.tolist())
        self.bioclim = check_bioclim(daset, metadata, state)
        self.nrasters = self.bioclim.shape[1]
        # finally, grab the files for reading the images
        self.imagekeys = daset[metadata.idCol].values
        self.filenames = daset[f'filepath_{year}'].values

    def __len__(self):
        return self.len_dset

    def check_idx(self, df_idx):
        return df_idx in self.index

    
    # idx should be a value from 0-N
    # where N is the length of the dataset
    # idx should not* be from the original
    # dataframe index
    
    def batch_datatype(self, idx, input_):
        if self.dataset_type is DatasetType.SINGLE_LABEL:
            return self.specs[idx], self.gens[idx], self.fams[idx], input_
        elif self.dataset_type is DatasetType.SINGLE_SPECIES:
            return self.all_specs_single[idx], self.all_gens_single[idx], self.all_fams_single[idx], input_
        elif self.dataset_type is DatasetType.MULTI_SPECIES:
            return self.all_specs_multi[idx], self.all_gens_multi[idx], self.all_fams_multi[idx], input_
    
    def __getitem__(self, idx):

        # bioclim only
        if self.datatype is DataType.BIOCLIM:
            # return labels + input data type
            return self.batch_datatype(idx, self.bioclim[idx])
        # naip only
        elif self.datatype is DataType.NAIP:
            fileHandle = np.load(f"{paths.IMAGES}{self.filenames[idx]}")
            img = fileHandle[f"{self.imagekeys[idx]}"]
            # scale+normalize image
            # NAIP imagery is 0-255 ints
            img = utils.scale(img, out_range=(0,1), min_=0, max_=255)
            img = TF.normalize(torch.tensor(img), self.mean, self.std)
            img = self.augment(img)
            return self.batch_datatype(idx, img)
        # both naip and bioclim
        elif self.datatype is DataType.JOINT_NAIP_BIOCLIM:
            fileHandle = np.load(f"{paths.IMAGES}{self.filenames[idx]}")
            img = fileHandle[f"{self.imagekeys[idx]}"]
            # scale+normalize image
            # NAIP imagery is 0-255 ints
            img = utils.scale(img, out_range=(0,1), min_=0, max_=255)
            img = TF.normalize(torch.tensor(img), self.mean, self.std)
            img = self.augment(img)
            return self.batch_datatype(idx, (img, self.bioclim[idx]))
