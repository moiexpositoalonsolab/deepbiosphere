# GIS packages
import geopandas as gpd
from pyproj import Proj
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, box
from scipy.spatial import cKDTree
import rasterio
from torch.utils.data import Dataset
# deepbiosphere packages
import deepbiosphere.NAIP_Utils  as naip
import deepbiosphere.Utils as utils
from deepbiosphere.Utils import paths
from rasterio.mask import mask
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import numpy as np
import pandas as pd
# miscellaneous packages
import os
import copy
from tqdm import tqdm
import random
import time
import math
import glob
import json



## ---------- MAGIC NUMBERS ---------- ##
# image size when fivecropping
FC_SIZE = 128
# possible rotation options for transform augmentations
DEGS = [15,30,45, 60,-75, 90]
# the kernel to use for gaussian blurring and sharpening
GKS = 5


# TODO: move shapefile into a method w/ state passed in
def get_bioclim_rasters(normalized='normalize', base_dir=paths.RASTERS, ras_paths=None, crs=naip.NAIP_CRS, out_range=(-1,1),outline=f"{paths.SHPFILES}gadm36_USA/gadm36_USA_1.shp"):

    # get outline of us
    us1 = gpd.read_file(outline) # the state's shapefiles
    # only going to use California
    ca = us1[us1.NAME_1 == 'California']
    # first, get raster files
    # always grabs standard WSG84 version which starts with b for bioclim
    if ras_paths is None:
        rasters = f"{paths.RASTERS}wc_30s_current/wc*bio_*.tif"
        ras_paths = glob.glob(rasters)
    if len(ras_paths) < 1:
        raise FileNotFoundError(f"no files found for {ras_paths}!")
    ras_agg = []
    # then load in rasters
    transfs = []
    for raster in tqdm(ras_paths, total=len(ras_paths), desc=" loading in rasters"):
        # load in raster
        src = rasterio.open(raster)
        # got to make sure it's all in the same crs
        # or indexing won't work
        assert str(src.crs) == crs; "CRS doesn't match!"
        ca = ca.to_crs(src.crs)
        cropped, transf = mask(src, ca.geometry, crop=True, pad=True,all_touched=True)
        masked = np.ma.masked_array(cropped, mask=(cropped==cropped.min()))
        transfs.append(transf)

        # depending on the chosen normalization strategy, normalize data

        if normalized == 'normalize':
            # z = (x- mean)/std
            masked = (masked - masked.mean()) / np.std(masked)
        elif normalized == 'min_max':
            masked = utils.scale(masked, out_range=out_range)
        elif normalized == 'none':
            pass
        else:
            raise NotImplementedError(f"No normalization for {normalized} implemented!")
        ras_agg.append((masked, transf))
    # finally, make sure that all the rasters are the same transform!
    for i, t1 in enumerate(transfs):
        for j, t2 in enumerate(transfs):
            assert t1 == t2, f"rasters don't match for {i}, {j} bioclim!"
    # returns a list of numpy arrays with each raster
    # plus the transform per-raster in order to use them together
    return ras_agg


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


# gross but pandas stores lists in csvs
# as strings, so need to unread that string
def parse_string_to_tuple(string):
    return eval(string)

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






class DeepbioDataset(Dataset):

    def __init__(self, dataset_name, datatype, dataset_type, state, year, band, split, latName, loName, idCol, augment, outline=f"{paths.SHPFILES}gadm36_USA/gadm36_USA_1.shp"):

        print("reading in data")
        # load in observations & metadata
        daset = pd.read_csv(f"{paths.OCCS}{dataset_name}.csv")        
        with open(f"{paths.OCCS}{dataset_name}_metadata.json", 'r') as f:
            metadata = json.load(f)
        pts = [Point(lon, lat) for lon, lat in zip(daset[loName], daset[latName])]
        # GBIF returns coordinates in WGS84 according to the API
        # https://www.gbif.org/article/5i3CQEZ6DuWiycgMaaakCo/gbif-infrastructure-data-processing
        daset = gpd.GeoDataFrame(daset, geometry=pts, crs=naip.NAIP_CRS)
        # pandas saves lists as strings in csv, gotta parse back to strings
        parsed = [parse_string_to_int(s) for s in daset.specs_overlap_id]
        daset['specs_overlap_id'] = parsed
        parsed = [parse_string_to_int(s) for s in daset.gens_overlap_id]
        daset['gens_overlap_id'] = parsed
        parsed = [parse_string_to_int(s) for s in daset.fams_overlap_id]
        daset['fams_overlap_id'] = parsed
        # for when using bioclim data, read in the bioclim rasters 
        self.rasters = get_bioclim_rasters()
        self.nrasters = len(self.rasters)


        # every species in dataset
        # is guaranteed to be in the species
        # column so this is chill for now
        self.nspec = len(metadata['spec_2_id'])
        self.ngen = len(metadata['gen_2_id'])
        self.nfam =  len(metadata['fam_2_id'])
        self.metadata = metadata
        self.datatype = datatype
        self.dataset_type = dataset_type
        self.total_len = len(daset)
        # for when using remote sensing data, read in NAIP statistics
        self.mean = metadata['dataset_means'][f"naip_{year}"]['means']
        self.std = metadata['dataset_means'][f"naip_{year}"]['stds']
        # only relevant for cases where remote sensing data used
        self.augment = augment
        # split data if using a test split
        if split != 'all_points':
            # if band is >=0, means to use the banding split
            if band >=0 :
                # save either the test or train split of the data
                daset =  daset[daset[f"{split}_{band}"]]
            # if band = -1, then use the uniform spatial split
            else:
                # save either the test or train split of the data
                daset = daset[daset.unif_train_test == split]
        daset = daset.to_crs(naip.NAIP_CRS)
        self.dataset = daset
        print(f"{split} dataset has {len(self.dataset)} points")
        # next, map the indices and save them as numpy
        self.idx_map = {
            k:v for k, v in
            zip(np.arange(len(self.dataset.index)), self.dataset.index)
        }
        # handle various cases of dataset_type
        # if training only on the specific species in the image
        if self.dataset_type == 'single_species':
            all_specs, all_gens, all_fams = [], [], []
            for spids, gids, fids in zip(daset.species_id, daset.genus_id, daset.family_id):
                specs_tens = np.full((self.nspec), 0)
                specs_tens[spids] += 1
                all_specs.append(specs_tens)
                gens_tens = np.full((self.ngen), 0)
                gens_tens[gids] += 1
                all_gens.append(gens_tens)
                fams_tens = np.full((self.nfam), 0)
                fams_tens[fids] += 1
                all_fams.append(fams_tens)
            self.all_specs = torch.tensor(np.stack(all_specs))
            self.all_gens = torch.tensor(np.stack(all_gens))
            self.all_fams = torch.tensor(np.stack(all_fams))
        else:
            # otherwise, load in the overlapping obs as well
            all_specs, all_gens, all_fams = [], [], []
            for spids, gids, fids in zip(daset.specs_overlap_id, daset.gens_overlap_id, daset.fams_overlap_id):
                specs_tens = np.full((self.nspec), 0)
                specs_tens[spids] += 1
                all_specs.append(specs_tens)
                gens_tens = np.full((self.ngen), 0)
                gens_tens[gids] += 1
                all_gens.append(gens_tens)
                fams_tens = np.full((self.nfam), 0)
                fams_tens[fids] += 1
                all_fams.append(fams_tens)
            self.all_specs = torch.tensor(np.stack(all_specs))
            self.all_gens = torch.tensor(np.stack(all_gens))
            self.all_fams = torch.tensor(np.stack(all_fams))
        # finally, we'll precompute the onehots for testing to save on loading costs
        self.specs = torch.tensor(daset.species_id.tolist())
        self.gens = torch.tensor(daset.genus_id.tolist())
        self.fams = torch.tensor(daset.family_id.tolist())
        self.index = daset.index
        # only compute when really needed, since it's super slow to load in
        if 'bioclim' in self.datatype:
            #  precompute  the bioclim variables at each test locatoin
            bioclim = []
            for point in tqdm(daset.geometry,total=len(daset), unit='point'):
                curr_bio = []
                # since we've confirmed all the rasters have identical
                # transforms, can just calculate the x,y coord once 
                x,y = rasterio.transform.rowcol(self.rasters[0][1], *point.xy)
                for j, (ras, transf) in enumerate(self.rasters):
                    curr_bio.append(ras[0,x,y])
                bioclim.append(curr_bio)
            self.bioclim = torch.tensor(np.squeeze(np.stack(bioclim)))
        # finally, grab the files for reading the images
        self.imagekeys = daset[idCol].values
        self.filenames = daset[f'filepath_{year}'].values
            
    def __len__(self):
        return len(self.dataset)

    def check_idx(df_idx):
        return df_idx in self.index
    # idx should be a value from 0-N
    # where N is the length of the dataset
    # idx should not* be from the original
    # dataframe index
    def __getitem__(self, idx):
        
        if self.datatype == 'bioclim':
            if self.dataset_type == 'single_label':
                return self.specs[idx], self.gens[idx], self.fams[idx], self.bioclim[idx]
            else: 
                return self.all_specs[idx], self.all_gens[idx], self.all_fams[idx], self.bioclim[idx]
        elif self.datatype == 'joint_naip_bioclim':
            fileHandle = np.load(f"{paths.IMAGES}{self.filenames[idx]}")
            img = fileHandle[f"{self.imagekeys[idx]}"]
            # scale+normalize image
            # NAIP imagery is 0-255 ints
            img = utils.scale(img, out_range=(0,1), min_=0, max_=255)
            img = TF.normalize(torch.tensor(img), self.mean, self.std)

            # add random augmentations
            if self.augment == 'random':
            # get fivecrop of image and randomly sample one of the images as the crop
            # don't use scaling! Just do the 128 pixels
                img = random_augment(img)
            elif self.augment == 'fivecrop':
                imgs  = TF.five_crop(torch.tensor(img), size=(FC_SIZE,FC_SIZE))
                which = random.sample(range(len(imgs)), 1)
                img = imgs[which[0]]
            # handle whether training with neighor labels or just individual 
            if self.dataset_type == 'single_label':
                return self.specs[idx], self.gens[idx], self.fams[idx], (img, self.bioclim[idx])
            else:
                return self.all_specs[idx], self.all_gens[idx], self.all_fams[idx], (img, self.bioclim[idx])
        else: 
            fileHandle = np.load(f"{paths.IMAGES}{self.filenames[idx]}")
            img = fileHandle[f"{self.imagekeys[idx]}"]
            # scale+normalize image
            # NAIP imagery is 0-255 ints
            img = utils.scale(img, out_range=(0,1), min_=0, max_=255)
            img = TF.normalize(torch.tensor(img), self.mean, self.std)

            # add random augmentations
            if self.augment == 'random':
            # get fivecrop of image and randomly sample one of the images as the crop
            # don't use scaling! Just do the 128 pixels
                img = random_augment(img)
            elif self.augment == 'fivecrop':
                imgs  = TF.five_crop(torch.tensor(img), size=(FC_SIZE,FC_SIZE))
                which = random.sample(range(len(imgs)), 1)
                img = imgs[which[0]]
            if self.dataset_type == 'single_label':
                return self.specs[idx], self.gens[idx], self.fams[idx], img
            else:
                return self.all_specs[idx], self.all_gens[idx], self.all_fams[idx], img
  








# TODO: remove below after testing code to make sure all is well




# class BioclimNAIP(Dataset):

#     def __init__(self, dataset_name, datatype, state, year, band, split, latName, loName, idCol, augment, outline=f"{paths.SHPFILES}gadm36_USA/gadm36_USA_1.shp"):

#         print("reading in data")
#         # load in observations & metadata
#         daset = pd.read_csv(f"{paths.OCCS}{dataset_name}.csv")
#         with open(f"{paths.OCCS}{dataset_name}_metadata.json", 'r') as f:
#             metadata = json.load(f)
#         pts = [Point(lon, lat) for lon, lat in zip(daset[loName], daset[latName])]
#         # GBIF returns coordinates in WGS84 according to the API
#         # https://www.gbif.org/article/5i3CQEZ6DuWiycgMaaakCo/gbif-infrastructure-data-processing
#         daset = gpd.GeoDataFrame(daset, geometry=pts, crs=naip.NAIP_CRS)
#         self.rasters = get_bioclim_rasters()
#         self.nrasters = len(self.rasters)
#         parsed = [parse_string_to_int(s) for s in daset.specs_overlap_id]
#         daset['specs_overlap_id'] = parsed
#         parsed = [parse_string_to_int(s) for s in daset.gens_overlap_id]
#         daset['gens_overlap_id'] = parsed
#         parsed = [parse_string_to_int(s) for s in daset.fams_overlap_id]
#         daset['fams_overlap_id'] = parsed



#         # every species in dataset
#         # is guaranteed to be in the species
#         # column so this is chill for now
#         self.nspec = len(metadata['spec_2_id'])
#         self.ngen = len(metadata['gen_2_id'])
#         self.nfam =  len(metadata['fam_2_id'])
#         self.metadata = metadata
#         self.datatype = datatype
#         self.total_len = len(daset)
#         self.mean = metadata['dataset_means'][f"naip_{year}"]['means']
#         self.std = metadata['dataset_means'][f"naip_{year}"]['stds']
#         self.augment = augment
#         # split data if using a test split
#         if split != 'all_points':
#             # if band is >=0, means to use the banding split
#             if band >=0 :
#                 # save either the test or train split of the data
#                 daset =  daset[daset[f"{split}_{band}"]]
#             # if band = -1, then use the uniform spatial split
#             else:
#                 # save either the test or train split of the data
#                 daset = daset[daset.unif_train_test == split]
#         daset = daset.to_crs(naip.NAIP_CRS)
#         self.dataset = daset
#         print(f"{split} dataset has {len(self.dataset)} points")
#         # next, map the indices and save them as numpy
#         self.idx_map = {
#             k:v for k, v in
#             zip(np.arange(len(self.dataset.index)), self.dataset.index)
#         }
#         # handle various cases of datatype
#         # if training only on the specific species in the image
#         if self.datatype == 'single_species':
#             all_specs, all_gens, all_fams = [], [], []
#             for spids, gids, fids in zip(daset.species_id, daset.genus_id, daset.family_id):
#                 specs_tens = np.full((self.nspec), 0)
#                 specs_tens[spids] += 1
#                 all_specs.append(specs_tens)
#                 gens_tens = np.full((self.ngen), 0)
#                 gens_tens[gids] += 1
#                 all_gens.append(gens_tens)
#                 fams_tens = np.full((self.nfam), 0)
#                 fams_tens[fids] += 1
#                 all_fams.append(fams_tens)
#             self.all_specs = torch.tensor(np.stack(all_specs))
#             self.all_gens = torch.tensor(np.stack(all_gens))
#             self.all_fams = torch.tensor(np.stack(all_fams))
#         else:
#             # otherwise, load in the overlapping obs as well
#             all_specs, all_gens, all_fams = [], [], []
#             for spids, gids, fids in zip(daset.specs_overlap_id, daset.gens_overlap_id, daset.fams_overlap_id):
#                 specs_tens = np.full((self.nspec), 0)
#                 specs_tens[spids] += 1
#                 all_specs.append(specs_tens)
#                 gens_tens = np.full((self.ngen), 0)
#                 gens_tens[gids] += 1
#                 all_gens.append(gens_tens)
#                 fams_tens = np.full((self.nfam), 0)
#                 fams_tens[fids] += 1
#                 all_fams.append(fams_tens)
#             self.all_specs = torch.tensor(np.stack(all_specs))
#             self.all_gens = torch.tensor(np.stack(all_gens))
#             self.all_fams = torch.tensor(np.stack(all_fams))
#         # finally, we'll precompute the onehots for testing to save on loading costs
#         self.specs = torch.tensor(daset.species_id.tolist())
#         self.gens = torch.tensor(daset.genus_id.tolist())
#         self.fams = torch.tensor(daset.family_id.tolist())
#         self.index = daset.index
#         #  precompute  the bioclim variables at each test locatoin
#         bioclim = []
#         for point in tqdm(daset.geometry,total=len(daset), unit='point'):
#             curr_bio = []
#             # since we've confirmed all the rasters have identical
#             # transforms, can just calculate the x,y coord once 
#             x,y = rasterio.transform.rowcol(self.rasters[0][1], *point.xy)
#             for j, (ras, transf) in enumerate(self.rasters):
#                 curr_bio.append(ras[0,x,y])
#             bioclim.append(curr_bio)
#         self.bioclim = torch.tensor(np.squeeze(np.stack(bioclim)))
#         # finally, grab the files for reading the images
#         self.imagekeys = daset[idCol].values
#         self.filenames = daset[f'filepath_{year}'].values
            
#     def __len__(self):
#         return len(self.dataset)

#     def check_idx(df_idx):
#         return df_idx in self.index
#     # idx should be a value from 0-N
#     # where N is the length of the dataset
#     # idx should not* be from the original
#     # dataframe index
#     def __getitem__(self, idx):
        
                
#         fileHandle = np.load(f"{paths.IMAGES}{self.filenames[idx]}")
#         img = fileHandle[f"{self.imagekeys[idx]}"]
#         # scale+normalize image
#         # NAIP imagery is 0-255 ints
#         img = utils.scale(img, out_range=(0,1), min_=0, max_=255)
#         img = TF.normalize(torch.tensor(img), self.mean, self.std)
        
#         # add random augmentations
#         if self.augment == 'random':
#         # get fivecrop of image and randomly sample one of the images as the crop
#         # don't use scaling! Just do the 128 pixels
#             img = random_augment(img)
#         elif self.augment == 'fivecrop':
#             imgs  = TF.five_crop(torch.tensor(img), size=(FC_SIZE,FC_SIZE))
#             which = random.sample(range(len(imgs)), 1)
#             img = imgs[which[0]]
#        	# handle whether training with neighor labels or just individual 
#         if self.datatype == 'single_label':
#             return self.specs[idx], self.gens[idx], self.fams[idx], (img, self.bioclim[idx])
#         else:
#             return self.all_specs[idx], self.all_gens[idx], self.all_fams[idx], (img, self.bioclim[idx])


# class Bioclim(Dataset):
#     def __init__(self, dataset_name, datatype, state, year, band, split, latName, loName, idCol, augment, outline=f"{paths.SHPFILES}gadm36_USA/gadm36_USA_1.shp"):

#         print("reading in data")
#         # load in observations & metadata
#         daset = pd.read_csv(f"{paths.OCCS}{dataset_name}.csv")
#         with open(f"{paths.OCCS}{dataset_name}_metadata.json", 'r') as f:
#             metadata = json.load(f)
#         pts = [Point(lon, lat) for lon, lat in zip(daset[loName], daset[latName])]
#         # GBIF returns coordinates in WGS84 according to the API
#         # https://www.gbif.org/article/5i3CQEZ6DuWiycgMaaakCo/gbif-infrastructure-data-processing
#         daset = gpd.GeoDataFrame(daset, geometry=pts, crs=naip.NAIP_CRS)
#         self.rasters = get_bioclim_rasters()
#         self.nrasters = len(self.rasters)
#         parsed = [parse_string_to_int(s) for s in daset.specs_overlap_id]
#         daset['specs_overlap_id'] = parsed
#         parsed = [parse_string_to_int(s) for s in daset.gens_overlap_id]
#         daset['gens_overlap_id'] = parsed
#         parsed = [parse_string_to_int(s) for s in daset.fams_overlap_id]
#         daset['fams_overlap_id'] = parsed
#         # every species in dataset
#         # is guaranteed to be in the species
#         # column so this is chill for now
#         self.nspec = len(metadata['spec_2_id'])
#         self.ngen = len(metadata['gen_2_id'])
#         self.nfam =  len(metadata['fam_2_id'])
#         self.metadata = metadata
#         self.datatype = datatype
#         self.total_len = len(daset)
#         if split != 'all_points':
#             # if band is >=0, means to use the banding split
#             if band >=0 :
#                 # save either the test or train split of the data
#                 daset =  daset[daset[f"{split}_{band}"]]
#             # if band = -1, then use the uniform spatial split
#             else:
#                 # save either the test or train split of the data
#                 daset = daset[daset.unif_train_test == split]
#         daset = daset.to_crs(naip.NAIP_CRS)
#         self.dataset = daset
#         print(f"{split} dataset has {len(self.dataset)} points")
#         # next, map the indices and save them as numpy
#         self.idx_map = {
#             k:v for k, v in
#             zip(np.arange(len(self.dataset.index)), self.dataset.index)
#         }
#         # TODO: turn this loading into a separate function called by each combo of datasets
#         # handle various cases of datatype
#         # if training only on the specific species in the image
#         if self.datatype == 'single_species':
#             all_specs, all_gens, all_fams = [], [], []
#             for spids, gids, fids in zip(daset.species_id, daset.genus_id, daset.family_id):
#                 specs_tens = np.full((self.nspec), 0)
#                 specs_tens[spids] += 1
#                 all_specs.append(specs_tens)
#                 gens_tens = np.full((self.ngen), 0)
#                 gens_tens[gids] += 1
#                 all_gens.append(gens_tens)
#                 fams_tens = np.full((self.nfam), 0)
#                 fams_tens[fids] += 1
#                 all_fams.append(fams_tens)
#             self.all_specs = torch.tensor(np.stack(all_specs))
#             self.all_gens = torch.tensor(np.stack(all_gens))
#             self.all_fams = torch.tensor(np.stack(all_fams))
#         else:
#             # otherwise, load in the overlapping obs as well
#             all_specs, all_gens, all_fams = [], [], []
#             for spids, gids, fids in zip(daset.specs_overlap_id, daset.gens_overlap_id, daset.fams_overlap_id):
#                 specs_tens = np.full((self.nspec), 0)
#                 specs_tens[spids] += 1
#                 all_specs.append(specs_tens)
#                 gens_tens = np.full((self.ngen), 0)
#                 gens_tens[gids] += 1
#                 all_gens.append(gens_tens)
#                 fams_tens = np.full((self.nfam), 0)
#                 fams_tens[fids] += 1
#                 all_fams.append(fams_tens)
#             self.all_specs = torch.tensor(np.stack(all_specs))
#             self.all_gens = torch.tensor(np.stack(all_gens))
#             self.all_fams = torch.tensor(np.stack(all_fams))
#         self.specs = torch.tensor(daset.species_id.tolist())
#         self.gens = torch.tensor(daset.genus_id.tolist())
#         self.fams = torch.tensor(daset.family_id.tolist())
#         self.index = daset.index
#         # finally, precompute the bioclim variables at each test locatoin
#         bioclim = []
#         for point in tqdm(daset.geometry,total=len(daset), unit='point'):
#             curr_bio = []
#             # since we've confirmed all the rasters have identical
#             # transforms, can just calculate the x,y coord once 
#             x,y = rasterio.transform.rowcol(self.rasters[0][1], *point.xy)
#             for j, (ras, transf) in enumerate(self.rasters):
#                 curr_bio.append(ras[0,x,y])
#             bioclim.append(curr_bio)
#         self.bioclim = torch.tensor(np.squeeze(np.stack(bioclim)))
            
            
#     def __len__(self):
#         return len(self.dataset)

#     def check_idx(df_idx):
#         return df_idx in self.index
#     # idx should be a value from 0-N
#     # where N is the length of the dataset
#     # idx should not* be from the original
#     # dataframe index
#     def __getitem__(self, idx):
        
#         if self.datatype == 'single_label':
#             return self.specs[idx], self.gens[idx], self.fams[idx], self.bioclim[idx]
#         else: 
#             return self.all_specs[idx], self.all_gens[idx], self.all_fams[idx], self.bioclim[idx]
        
        
  
# class NAIP(Dataset):

#     def __init__(self, dataset_name, datatype, state, year, band, split, latName, loName, idCol, augment):

#         print("reading in data")
#         # load in observations & metadata
#         daset = pd.read_csv(f"{paths.OCCS}{dataset_name}.csv")
#         with open(f"{paths.OCCS}{dataset_name}_metadata.json", 'r') as f:
#             metadata = json.load(f)
#         self.mean = metadata['dataset_means'][f"naip_{year}"]['means']
#         self.std = metadata['dataset_means'][f"naip_{year}"]['stds']
#         self.augment = augment

#         parsed = [parse_string_to_int(s) for s in daset.specs_overlap_id]
#         daset['specs_overlap_id'] = parsed
#         parsed = [parse_string_to_int(s) for s in daset.gens_overlap_id]
#         daset['gens_overlap_id'] = parsed
#         parsed = [parse_string_to_int(s) for s in daset.fams_overlap_id]
#         daset['fams_overlap_id'] = parsed

#         # every species in dataset
#         # is guaranteed to be in the species
#         # column so this is chill for now
#         self.nspec = len(metadata['spec_2_id'])
#         self.ngen = len(metadata['gen_2_id'])
#         self.nfam =  len(metadata['fam_2_id'])
#         self.metadata = metadata
#         self.datatype = datatype
#         self.band = band
#         self.total_len = len(daset)
#         # TODO: turn this into a function as well???
#         # only split if you don't want to train on all obs
#         if split != 'all_points':
#             # if band is >=0, means to use the banding split
#             if band >=0 :
#                 # save either the test or train split of the data
#                 daset =  daset[daset[f"{split}_{band}"]]
#             # if band = -1, then use the uniform spatial split
#             else:
#                 # save either the test or train split of the data
#             daset = daset[daset.unif_train_test == split]
#         self.dataset = daset
#         print(f"{split} dataset has {len(self.dataset)} points")
#         # next, map the indices and save them as numpy
#         self.idx_map = {
#             k:v for k, v in
#             zip(np.arange(len(self.dataset.index)), self.dataset.index)
#         }
#         # also be sure to save the ids of the observations
#         # for other downstream analyses
#         self.ids = daset[idCol]
#         # TODO: make a function
#         if self.datatype == 'single_species':
#              # we'll precompute the onehots for training to save on loading costs
#             all_specs, all_gens, all_fams = [], [], []
#             for spids, gids, fids in zip(daset.species_id, daset.genus_id, daset.family_id):
#                 specs_tens = np.full((self.nspec), 0)
#                 specs_tens[spids] += 1
#                 all_specs.append(specs_tens)
#                 gens_tens = np.full((self.ngen), 0)
#                 gens_tens[gids] += 1
#                 all_gens.append(gens_tens)
#                 fams_tens = np.full((self.nfam), 0)
#                 fams_tens[fids] += 1
#                 all_fams.append(fams_tens)
#             self.all_specs = torch.tensor(np.stack(all_specs))
#             self.all_gens = torch.tensor(np.stack(all_gens))
#             self.all_fams = torch.tensor(np.stack(all_fams))
#         else:
#             # we'll precompute the multi-hots for training to save on loading costs
#             all_specs, all_gens, all_fams = [], [], []
#             for spids, gids, fids in zip(daset.specs_overlap_id, daset.gens_overlap_id, daset.fams_overlap_id):
#                 specs_tens = np.full((self.nspec), 0)
#                 specs_tens[spids] += 1
#                 all_specs.append(specs_tens)
#                 gens_tens = np.full((self.ngen), 0)
#                 gens_tens[gids] += 1
#                 all_gens.append(gens_tens)
#                 fams_tens = np.full((self.nfam), 0)
#                 fams_tens[fids] += 1
#                 all_fams.append(fams_tens)
#             self.all_specs = torch.tensor(np.stack(all_specs))
#             self.all_gens = torch.tensor(np.stack(all_gens))
#             self.all_fams = torch.tensor(np.stack(all_fams))
#         # finally, we'll precompute the onehots for training to save on loading costs
#         self.specs = torch.tensor(daset.species_id.tolist())
#         self.gens = torch.tensor(daset.genus_id.tolist())
#         self.fams = torch.tensor(daset.family_id.tolist())
#         # we use the index of the dataset rather than 0-N
#         # mapping because it preserves the absolute indexing
#         # relative to the train / test split and allows us to confirm
#         # that the observations are indeed separate.
#         # I don't think dictionary indexing is that slow so it's not
#         # going to lead to substantial performance issues
#         self.index = daset.index
#         self.imagekeys = daset[idCol].values
#         self.filenames = daset[f'filepath_{year}'].values
        
#     def __len__(self):
#         return len(self.dataset)

#     def check_idx(df_idx):
#         return df_idx in self.index
#     # idx should be a value from 0-N
#     # where N is the length of the dataset
#     # idx should not* be from the original
#     # dataframe index
#     def __getitem__(self, idx):
        
#         fileHandle = np.load(f"{paths.IMAGES}{self.filenames[idx]}")
#         img = fileHandle[f"{self.imagekeys[idx]}"]
#         # scale+normalize image
#         # NAIP imagery is 0-255 ints
#         img = utils.scale(img, out_range=(0,1), min_=0, max_=255)
#         img = TF.normalize(torch.tensor(img), self.mean, self.std)
        
#         # add random augmentations
#         if self.augment == 'random':
#         # get fivecrop of image and randomly sample one of the images as the crop
#         # don't use scaling! Just do the 128 pixels
#             img = random_augment(img)
#         elif self.augment == 'fivecrop':
#             imgs  = TF.five_crop(torch.tensor(img), size=(FC_SIZE,FC_SIZE))
#             which = random.sample(range(len(imgs)), 1)
#             img = imgs[which[0]]
#         if self.datatype == 'single_label':
#             return self.specs[idx], self.gens[idx], self.fams[idx], img
#         else:
#             return self.all_specs[idx], self.all_gens[idx], self.all_fams[idx], img
