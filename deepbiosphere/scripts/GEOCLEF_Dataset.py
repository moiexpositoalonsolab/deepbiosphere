from deepbiosphere.scripts.GEOCLEF_Config import paths
import glob
from rasterio.mask import mask
#import geopandas as gpd
import geojson
from shapely.geometry import mapping
import rasterio
import math
import math
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from deepbiosphere.scripts import GEOCLEF_Utils as utils
from deepbiosphere.GLC.environmental_raster_glc import Raster, PatchExtractor

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


def get_gbif_data(pth, split, country, organism):
    ## Grab GBIF observation data

    obs_pth = "{}occurrences/occurrences_{}_{}_{}.csv".format(pth, country, organism, split)
    return pd.read_csv(obs_pth, sep=';')  

def get_gbif_rasters_data(pth, country, organism):
#     {pth}/occurrences/joint_obs_{region}{plant}_train_rasters.csv
    obs_pth = "{}occurrences/joint_obs_{}_{}_train_rasters.csv".format(pth, country, organism)
    joint_obs = pd.read_csv(obs_pth)  
    joint_obs.all_specs = joint_obs.all_specs.apply(lambda x: parse_string_to_int(x))
    joint_obs.all_gens = joint_obs.all_gens.apply(lambda x: parse_string_to_string(x))
    joint_obs.all_fams = joint_obs.all_fams.apply(lambda x: parse_string_to_string(x))
    joint_obs.lat_lon = joint_obs.lat_lon.apply(lambda x: parse_string_to_tuple(x))
    return joint_obs

def get_joint_gbif_data(pth, country, organism):
    ## Grab GBIF observation data
    obs_pth = "{}occurrences/joint_obs_{}_{}.csv".format(pth, country, organism)
    joint_obs = pd.read_csv(obs_pth)  
    joint_obs.all_specs = joint_obs.all_specs.apply(lambda x: parse_string_to_int(x))
    joint_obs.all_gens = joint_obs.all_gens.apply(lambda x: parse_string_to_string(x))
    joint_obs.all_fams = joint_obs.all_fams.apply(lambda x: parse_string_to_string(x))
    joint_obs.lat_lon = joint_obs.lat_lon.apply(lambda x: parse_string_to_tuple(x))
    return joint_obs
def xy_2_range_center(pix_res, x, y):
    half = pix_res /2
    xmin, xmax = x-half, x+half
    xmin, xmax = int(xmin), int(xmax)
    ymin, ymax = y-half, y+half
    ymin, ymax = int(ymin), int(ymax) 
    return xmin, xmax, ymin, ymax
        
def parse_string_to_tuple(string):
    return eval(string)
def parse_string_to_string(string):
    string = string.replace("{", '').replace("}", "").replace("'", '')
    split = string.split(", ")
    return split

def parse_string_to_int(string):
    string = string.replace("{", '').replace("}", "").replace("'", '')
    split = string.split(", ")
    return [int(s) for s in split]

def get_big_cali_shape(base_dir):
    path = '{}us_shapefiles/bigcali.geojson'.format(base_dir)
    with open(path) as f:
        geocali = geojson.load(f)
        return [geocali]

def get_cali_shape(base_dir):
    path = '{}us_shapefiles/districts/states/CA/shape.geojson'.format(base_dir)
    with open(path) as f:
            geocali = geojson.load(f)
    return [geocali]

def get_us_bioclim(base_dir):
    rasters = "{}rasters/bio*/*_{}.tif".format(base_dir, 'USA')
    ras_paths = glob.glob(rasters)
    return ras_paths

raster_metadata = {
    'bio_1': {'min_val': -116, 'max_val': 259, 'nan': -2147483647, 'new_nan': -117, 'mu': 101, 'sigma': 58},
    'bio_2': {'min_val': -53, 'max_val': 361, 'nan': -2147483647, 'new_nan': -54, 'mu': 131, 'sigma': 28},
    'bio_3': {'min_val': 19, 'max_val': 69, 'nan': -2147483647, 'new_nan': 18, 'mu': 36, 'sigma': 8},
    'bio_4': {'min_val': 1624, 'max_val': 13302, 'nan': -2147483647, 'new_nan': 1623, 'mu': 8267, 'sigma': 2152},
    'bio_5': {'min_val': -25, 'max_val': 457, 'nan': -2147483647, 'new_nan': -26, 'mu': 289, 'sigma': 48},
    'bio_6': {'min_val': -276, 'max_val': 183, 'nan': -2147483647, 'new_nan': -277, 'mu': -78, 'sigma': 83},
    'bio_7': {'min_val': 117, 'max_val': 515, 'nan': -2147483647, 'new_nan': 116, 'mu': 367, 'sigma': 72},
    'bio_8': {'min_val': -169, 'max_val': 332, 'nan': -2147483647, 'new_nan': -170, 'mu': 149, 'sigma': 82},
    'bio_9': {'min_val': -181, 'max_val': 331, 'nan': -2147483647, 'new_nan': -182, 'mu': 54, 'sigma': 114},
    'bio_10': {'min_val': -53, 'max_val': 361, 'nan': -2147483647, 'new_nan': -54, 'mu': 205, 'sigma': 47},
    'bio_11': {'min_val': -186, 'max_val': 220, 'nan': -2147483647, 'new_nan': -187, 'mu': -7, 'sigma': 80},
    'bio_12': {'min_val': -35, 'max_val': 3385, 'nan': -2147483647, 'new_nan': -36, 'mu': 746, 'sigma': 383},
    'bio_13': {'min_val': 7, 'max_val': 570, 'nan': -2147483647, 'new_nan': 6, 'mu': 98, 'sigma': 47},
    'bio_14': {'min_val': 0, 'max_val': 184, 'nan': -2147483647, 'new_nan': -1, 'mu': 34, 'sigma': 26},
    'bio_15': {'min_val': 5, 'max_val': 140, 'nan': -2147483647, 'new_nan': 4, 'mu': 38, 'sigma': 23},
    'bio_16': {'min_val': 19, 'max_val': 1546, 'nan': -2147483647, 'new_nan': 18, 'mu': 265, 'sigma': 132},
    'bio_17': {'min_val': 0, 'max_val': 612, 'nan': -2147483647, 'new_nan': -1, 'mu': 117, 'sigma': 84},
    'bio_18': {'min_val': 1, 'max_val': 777, 'nan': -2147483647, 'new_nan': 0, 'mu': 213, 'sigma': 107},
    'bio_19': {'min_val': 5, 'max_val': 1485, 'nan': -2147483647, 'new_nan': 4, 'mu': 163, 'sigma': 137},
}
nan = -2147483647

def open_raster(raster):
    ras_name = raster.split("/")[-1].split("_{}".format('USA'))[0]
    print("loading {}".format(ras_name))
    nan = raster_metadata[ras_name]['nan']
    src = rasterio.open(raster, nodata=nan)
    return src

def coord_2_index(affine, lat, lon):
    x, y =  ~affine * (lon, lat)
    return int(round(x)), int(round(y))
def latlon_2_index(affine, latlon):
    y, x =  ~affine * (latlon[1], latlon[0])
    return int(round(x)), int(round(y))

def raster_cnn_image(rasters, xmin, xmax, ymin, ymax, nan):
    if ymin < 0:
        # find how many ocean nans are missing
        diff = xmax-xmin
        # recreate the missing westernmost part of image
        extra = np.full((rasters.shape[0],diff, -ymin), nan)
        # get the parts of the raster that do exist
        inc_rasters = rasters[:,xmin:xmax,0:ymax]
        # and append the two
        env_rasters = np.concatenate([extra, inc_rasters], axis=2)
    # if the range of the rasters in other dimensions is out of bounds, then it's a dataset error and return
    
    elif xmin < 0 or xmax > rasters.shape[1] or ymax > rasters.shape[2]:
        print("riperoni")
        print(xmin, xmax, ymin, ymax, rasters.shape)
        import pdb; pdb.set_trace()
        exit(1), "{} is outside bounds of env raster image!".format(lat_lon)
    else: 
        env_rasters = rasters[:,xmin:xmax,ymin:ymax]
    return env_rasters

def raster_filter_2_cali(base_dir, obs):
    
    geoms = get_cali_shape(base_dir)
    ras_paths = get_us_bioclim(base_dir)
    src = open_raster(ras_paths[0])       
# filter down the dataset to only in-range observations
    filt_obs = filter_to_bioclim(obs, src, geoms, nan)
    return  filt_obs

def get_bioclim_rasters(base_dir, region, normalized, obs, big=True):
    
    if region ==  'cali':
        geoms = get_cali_shape(base_dir) if not big else get_big_cali_shape(base_dir)
        ras_paths = get_us_bioclim(base_dir)
    else:
        raise NotImplementedError
    ras_agg = []
    aff_agg = []
    for raster in ras_paths:
        if region == 'cali': 
            src = open_raster(raster) 
        else:
            raise NotImplementedError
        
        masked, affine = mask(src, geoms, nodata=nan, filled=False, crop=True, pad=True)
    #         z = (x- mean)/std
        if normalized == 'normalize':
            masked = (masked - masked.mean()) / masked.var()
        elif normalized == 'min_max':
            masked = utils.scale(masked)
        elif normalized == 'none':
            pass
        else:
            raise NotImplmentedError
            
        ras_agg.append(masked)
        # this method relies on all affine transformations being the same, need to change that for more environmental rasters
        aff_agg.append(affine)        
    # filter down the dataset to only in-range observations
    filt_obs = filter_to_bioclim(obs, src, geoms, nan)
    # make sure rasters in same affine 
    # replace nan value with something more reasonable
    if normalized == 'normalize':

        mins = [r.min() for r in ras_agg]
        maxs = [r.max() for r in ras_agg]
        min_ = max(maxs) - min(mins) * 2
        min_ = -min_
#         print('normalizing to ', -min_, max(maxs), min(mins))
        ras_agg = [r.filled(min_) for r in ras_agg]
        
    elif normalized == 'min_max':
#         print('min maxing')
        mins = [r.min() for r in ras_agg]
        maxs = [r.max() for r in ras_agg]
        min_ = max(maxs) - min(mins) * 2
        min_ = -min_
#         print('min maxing to ', -min_, max(maxs), min(mins))
        ras_agg = [r.filled(min_) for r in ras_agg]
    else:
        min_ = min([r.data.min() for r in ras_agg])
        ras_agg = [r.filled(min_) for r in ras_agg]
    
    for i in range(len(aff_agg)):
        for j in range(len(aff_agg)):
            assert aff_agg[i] == aff_agg[j], "not all affines are the same! please align all rasters to same affine transformation"
    # make sure rasters cropped to same range
    for i in range(len(ras_agg)):
        for j in range(len(ras_agg)):
            assert ras_agg[i].shape == ras_agg[j].shape, "rasters not cropped to the same size!"
    all_ras = np.stack(np.squeeze(ras_agg))
    affine = aff_agg[0] # can do because confirmed that all affines are the same, will need to change if having rasters of different affines
    
    return all_ras, affine, filt_obs, min_

def filter_to_bioclim(obs, src, geoms, nan):
    bad_ids = []
    masked, affine = mask(src, geoms, nodata=nan, filled=True, crop=False, pad=True, all_touched=True)
    masked = np.squeeze(masked)
    for i, ob in obs.iterrows():
        x, y = latlon_2_index(affine, ob.lat_lon)
#         print("x y: ", x, y, ob.lat_lon)
        if x >= masked.shape[0] or y >= masked.shape[1] or x < 0 or y < 0:
            bad_ids.append(ob.id)
            continue
        val = masked[x,y]
        if val == nan:
            bad_ids.append(ob.id)
    return obs[~obs.id.isin(bad_ids)]

def subpath_2_img(pth, subpath, id_):
    alt = "{}{}{}_alti.npy".format(pth, subpath, id_)
    rgbd = "{}{}{}.npy".format(pth, subpath, id_)    
    # Necessary because some data corrupted...
    np_al = np.load(alt)
    np_img = np.load(rgbd)
    np_al = np.expand_dims(np_al, 2)
    np_all = np.concatenate((np_al, np_img), axis=2)
    return np.transpose(np_all,(2, 0, 1))

def subpath_2_img_noalt(pth, subpath, id_):
    rgbd = "{}{}{}.npy".format(pth, subpath, id_)    
    # Necessary because some data corrupted...
    np_img = np.load(rgbd)
    np_img = np_img[:,:,:4]
    return np.transpose(np_img,(2, 0, 1))


def image_from_id(id_, pth, altitude=True):
    # make sure image and path are for same region
    cdd, ab, cd = utils.id_2_file(id_)
    subpath = "patches_{}/{}/{}/".format('fr', cd, ab) if id_ >= 10000000 else "patches_{}/patches_{}_{}/{}/{}/".format('us', 'us', cdd, cd, ab)
    return subpath_2_img(pth, subpath, id_) if altitude else subpath_2_img_noalt(pth, subpath, id_)

def freq_from_dict(f_dict):
    list(f_dict.items())
    # sort frequency list by species_id (key of dict)
    return [freq for (sp_id, freq) in sorted(list(f_dict.items()), key=lambda x:x[0])]    

def get_shapes(id_, pth, altitude):
    tens = image_from_id(id_, pth, altitude)
    # channels, alt_shape, rgbd_shape    
    return tens.shape[0], tens.shape[1], tens.shape[2]



def add_genus_family_data(pth, train):
    ## getting family, genus, species ids for each observation
    # get all relevant files
    gbif_meta = pd.read_csv("{}occurrences/species_metadata.csv".format(pth), sep=";")
    taxons = pd.read_csv("{}occurrences/Taxon.tsv".format(pth), sep="\t")
    # get all unique species ids in us train data
    us_celf_spec = train.species_id.unique()
    # get all the gbif species ids for all the species in the us sample
    conversion = gbif_meta[gbif_meta['species_id'].isin(us_celf_spec)]
    gbif_specs = conversion.GBIF_species_id.unique()
    # get dict that maps CELF id to GBIF id
    spec_2_gbif = dict(zip(conversion.species_id, conversion.GBIF_species_id))
    train['gbif_id'] = train['species_id'].map(spec_2_gbif)
    # grab all the phylogeny mappings from the gbif taxons file for all the given species
    # GBIF id == taxonID
    taxa = taxons[taxons['taxonID'].isin(gbif_specs)]
    phylogeny = taxa[['taxonID', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus']]
    gbif_2_fam = dict(zip(phylogeny.taxonID, phylogeny.family))
    gbif_2_gen = dict(zip(phylogeny.taxonID, phylogeny.genus))
    train['family'] = train['gbif_id'].map(gbif_2_fam)
    train['genus'] = train['gbif_id'].map(gbif_2_gen)
    return train


def map_key_2_index(df, key):
    key_2_id = {
        k:v for k, v in 
        zip(df[key].unique(), np.arange(len(df[key].unique())))
    }
    df[key] = df[key].map(key_2_id)
    return df

def dict_key_2_index(df, key):
    return {
        k:v for k, v in 
        zip(df[key].unique(), np.arange(len(df[key].unique())))
    }


def prep_data(obs):

    spec_dict = dict_key_2_index(obs, 'species_id')
    inv_spec = {v: k for k, v in spec_dict.items()}
    obs = map_key_2_index(obs, 'species_id')
    obs = map_key_2_index(obs, 'genus')
    obs = map_key_2_index(obs, 'family')

    return obs, inv_spec    
    
# TODO: assumes that species_id, genus, family columns contain all possible values contained in extra_obs    
def prep_joint_data(obs):
    spec_dict = dict_key_2_index(obs, 'species_id')
    gen_dict = dict_key_2_index(obs, 'genus')
    fam_dict = dict_key_2_index(obs, 'family')
    inv_spec = {v: k for k, v in spec_dict.items()}
    # for each set in
    obs = obs.assign(all_specs=[[spec_dict[k] for k in row ] for row in obs.all_specs])
    obs = obs.assign(all_gens=[[gen_dict[k] for k in row ] for row in obs.all_gens])
    obs = obs.assign(all_fams=[[fam_dict[k] for k in row ] for row in obs.all_fams])    
    return obs, inv_spec      

def prep_joint_data_toy(obs):
    spec_dict = dict_key_2_index(obs, 'species_id')
    gen_dict = dict_key_2_index(obs, 'genus')
    fam_dict = dict_key_2_index(obs, 'family')
    inv_spec = {v: k for k, v in spec_dict.items()}
    inv_gen = {v: k for k, v in gen_dict.items()}
    inv_fam = {v: k for k, v in fam_dict.items()}    
    # for each set in
    obs = obs.assign(all_specs=[[spec_dict[k] for k in row ] for row in obs.all_specs])
    obs = obs.assign(all_gens=[[gen_dict[k] for k in row ] for row in obs.all_gens])
    obs = obs.assign(all_fams=[[fam_dict[k] for k in row ] for row in obs.all_fams])    
    return obs, inv_spec, inv_gen, inv_fam

#TODO: normalize dataset range to gaussian distribution
def normalize_dataset():
    pass

class GEOCELF_Dataset(Dataset):
    def __init__(self, base_dir, organism, country='us', altitude=True, transform=None):

        self.base_dir = base_dir
        self.country = country
        self.organism = organism
        self.split = 'train'
        self.altitude = altitude
        obs = get_gbif_data(self.base_dir, self.split, country, organism)
        obs.fillna('nan', inplace=True)
        obs = add_genus_family_data(self.base_dir, obs)
        obs, inv_spec  = prep_data(obs)
        self.idx_2_id = inv_spec
        # Grab only obs id, species id, genus, family because lat /lon not necessary at the moment
        self.num_specs = len(obs.species_id.unique())
        self.num_fams = len(obs.family.unique())
        self.num_gens = len(obs.genus.unique())
        self.spec_freqs = obs.species_id.value_counts().to_dict()
        self.gen_freqs = obs.genus.value_counts().to_dict()
        self.fam_freqs = obs.family.value_counts().to_dict()                
        # convert to numpy
        self.obs = obs[['id', 'species_id', 'genus', 'family']].values
        self.transform = transform
        channels, alt_shape, rgbd_shape = get_shapes(self.obs[0,0], self.base_dir, self.altitude)
        self.channels = channels
        self.alt_shape = alt_shape
        self.rgbd_shape = rgbd_shape


    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # obs is of shape [id, species_id, genus, family]    
        id_ = self.obs[idx, 0]
        images = image_from_id(id_, self.base_dir, self.altitude)
        composite_label = self.obs[idx, 1:] # get genus, family as well
        if self.transform:
            images = self.transform(images)
        return (composite_label, images)


class GEOCELF_Dataset_Full(Dataset):
    def __init__(self, base_dir, organism, altitude=True, transform=None):

        self.base_dir = base_dir
        self.split = 'train'
        us_obs = get_gbif_data(self.base_dir, self.split, 'us', organism)
        fr_obs = get_gbif_data(self.base_dir, self.split, 'fr', organism)
        self.altitude = altitude
        obs = pd.concat([us_obs, fr_obs])
    
        
        obs.fillna('nan', inplace=True)
        obs = add_genus_family_data(self.base_dir, obs)
        obs, inv_spec  = prep_data(obs)
        self.idx_2_id = inv_spec
        # Grab only obs id, species id, genus, family because lat /lon not necessary at the moment
        self.num_specs = len(obs.species_id.unique())
        self.num_fams = len(obs.family.unique())
        self.num_gens = len(obs.genus.unique())
        self.spec_freqs = obs.species_id.value_counts().to_dict()
        self.gen_freqs = obs.genus.value_counts().to_dict()
        self.fam_freqs = obs.family.value_counts().to_dict()                
        self.obs = obs[['id', 'species_id', 'genus', 'family']].values
        self.transform = transform
        channels, alt_shape, rgbd_shape = get_shapes(self.obs[0,0], self.base_dir, self.altitude)
        self.channels = channels
        self.alt_shape = alt_shape
        self.rgbd_shape = rgbd_shape

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # obs is of shape [id, species_id, genus, family]    
        id_, label = self.obs[idx, 0], self.obs[idx, 1]
        images = image_from_id(id_, self.base_dir, self.altitude)
        composite_label = self.obs[idx, 1:] # get genus, family as well
        if self.transform:
            images = self.transform(images)
        return (composite_label, images)    
    
class GEOCELF_Test_Dataset(Dataset):
    def __init__(self, base_dir, organism, country='us', transform=None):
        
        self.base_dir = base_dir
        self.country = country
        self.split = 'test'
        self.altitude = altitude        
        obs = get_gbif_data(self.base_dir, self.split, country, organism)
        self.obs = obs[['id']].values
        _, alt_shape, rgbd_shape = get_shapes(self.obs[0, 0], self.base_dir, self.altitude)
        self.rgbd_shape = rgbd_shape
        self.transform = transform

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # obs is of shape [id, species_id, genus, family]    
        id_ = self.obs[idx,0]
        images = image_from_id(id_, self.base_dir, self.altitude)        
        if self.transform:
            images = self.transform(images)
        return (images, id_)    

class GEOCELF_Test_Dataset_Full(Dataset):
    def __init__(self, base_dir, organism, transform=None):
        
        self.base_dir = base_dir
        self.split = 'test'
        us_obs = get_gbif_data(self.base_dir, self.split, 'us', organism)
        fr_obs = get_gbif_data(self.base_dir, self.split, 'fr', organism)
        obs = pd.concat([us_obs, fr_obs])
        self.altitude = altitude        
        self.obs = obs[['id']].values
        _, alt_shape, rgbd_shape = get_shapes(self.obs[0, 0], self.base_dir, self.altitude)
        self.alt_shape = alt_shape
        self.rgbd_shape = rgbd_shape
        self.transform = transform
#        self.obs = obs[['id']].to_numpy()
        self.transform = transform

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # obs is of shape [id, species_id, genus, family]    
        id_ = self.obs[idx,0]
        images = image_from_id(id_, self.base_dir, self.altitude)               
        if self.transform:
            images = self.transform(images)
        return (images, id_)    
    
    
class GEOCELF_Dataset_Joint(Dataset):
    def __init__(self, base_dir, organism, country='us', altitude=True, transform=None):
        self.base_dir = base_dir
        self.country = country
        self.altitude = altitude        
        self.organism = organism
        obs = get_joint_gbif_data(self.base_dir, country, organism)
        obs.fillna('nan', inplace=True)        
        obs, inv_spec = prep_joint_data(obs)
        self.idx_2_id = inv_spec
        # Grab only obs id, species id, genus, family because lat /lon not necessary at the moment
        self.num_specs = len(obs.species_id.unique())
        self.num_fams = len(obs.family.unique())
        self.num_gens = len(obs.genus.unique())
        self.spec_freqs = obs.species_id.value_counts().to_dict()
        self.gen_freqs = obs.genus.value_counts().to_dict()
        self.fam_freqs = obs.family.value_counts().to_dict()                
        self.obs = obs[['id', 'all_specs', 'all_fams', 'all_gens']].values
        self.transform = transform
        channels, alt_shape, rgbd_shape = get_shapes(self.obs[0,0], self.base_dir, self.altitude)
        self.channels = channels
        self.alt_shape = alt_shape
        self.rgbd_shape = rgbd_shape
    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # obs is of shape [id, species_id, genus, family]    
        id_ = self.obs[idx, 0]
        images = image_from_id(id_, self.base_dir, self.altitude)                    
        specs_label = self.obs[idx, 1]
        gens_label = self.obs[idx, 3]
        fams_label = self.obs[idx, 2]        
        if self.transform:
            images = self.transform(images)
        return (specs_label, gens_label, fams_label, images)    
    

class GEOCELF_Dataset_Joint_Full(Dataset):
    def __init__(self, base_dir, organism, altitude=True, transform=None):
        
        self.base_dir = base_dir
        self.split = 'train'
        self.organism = organism
        self.altitude = altitude        
        us_obs = get_joint_gbif_data(self.base_dir, 'us', organism)
        fr_obs = get_joint_gbif_data(self.base_dir, 'fr', organism)
        obs = pd.concat([us_obs, fr_obs])
        obs.fillna('nan', inplace=True)        
        obs, inv_spec = prep_joint_data(obs)
        self.idx_2_id = inv_spec
        # Grab only obs id, species id, genus, family because lat /lon not necessary at the moment
        self.num_specs = len(obs.species_id.unique())
        self.num_fams = len(obs.family.unique())
        self.num_gens = len(obs.genus.unique())
        self.spec_freqs = obs.species_id.value_counts().to_dict()
        self.gen_freqs = obs.genus.value_counts().to_dict()
        self.fam_freqs = obs.family.value_counts().to_dict()                
        self.obs = obs[['id', 'all_specs', 'all_fams', 'all_gens']].values
        self.transform = transform
        channels, alt_shape, rgbd_shape = get_shapes(self.obs[0,0], self.base_dir, self.altitude)
        self.channels = channels
        self.alt_shape = alt_shape
        self.rgbd_shape = rgbd_shape
    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()
        id_ = self.obs[idx, 0]            
        # obs is of shape [id, species_id, genus, family]    
        id_, label = self.obs[idx, 0], self.obs[idx, 1]
        images = image_from_id(id_, self.base_dir, self.altitude)            
        specs_label = self.obs[idx, 1]
        gens_label = self.obs[idx, 3]
        fams_label = self.obs[idx, 2]        
        if self.transform:
            images = self.transform(images)
        return (specs_label, gens_label, fams_label, images)  

    
class Joint_Toy_Dataset(Dataset):
    def __init__(self, base_dir, organism, country='us', transform=None):
        self.base_dir = base_dir
        self.country = country
        self.organism = organism
        self.altitude = altitude        
        obs = get_joint_gbif_data(self.base_dir, country, organism)
        obs.fillna('nan', inplace=True)        
        obs, inv_spec, inv_gen, inv_fam = prep_joint_data_toy(obs)
        self.spec_idx_2_id = inv_spec
        self.gen_idx_2_id = inv_gen
        self.fam_idx_2_id = inv_fam        
        print(type(self.spec_idx_2_id))
        # Grab only obs id, species id, genus, family because lat /lon not necessary at the moment
        self.num_specs = len(obs.species_id.unique())
        self.num_fams = len(obs.family.unique())
        self.num_gens = len(obs.genus.unique())
        self.spec_freqs = obs.species_id.value_counts().to_dict()
        self.gen_freqs = obs.genus.value_counts().to_dict()
        self.fam_freqs = obs.family.value_counts().to_dict()  
        obs = obs[:500]
        self.obs = obs[['id', 'all_specs', 'all_fams', 'all_gens']].values
        self.transform = transform
        channels, alt_shape, rgbd_shape = get_shapes(self.obs[0,0], self.base_dir, self.altitude)
        self.channels = channels
        self.alt_shape = alt_shape
        self.rgbd_shape = rgbd_shape
        
    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # obs is of shape [id, species_id, genus, family]    
        id_ = self.obs[idx, 0]
        images = image_from_id(id_, self.base_dir, self.altitude)                    
        specs_label = self.obs[idx, 1]
        gens_label = self.obs[idx, 3]
        fams_label = self.obs[idx, 2]        
        if self.transform:
            images = self.transform(images)
        return (specs_label, gens_label, fams_label, images)   
    
    
class Single_Toy_Dataset(Dataset):
    def __init__(self, base_dir, organism, country='us', transform=None):

        self.base_dir = base_dir
        self.country = country
        self.organism = organism
        self.split = 'train'
        obs = get_gbif_data(self.base_dir, self.split, country, organism)
        obs.fillna('nan', inplace=True)
        obs = add_genus_family_data(self.base_dir, obs)
        obs, inv_spec  = prep_data(obs)
        self.idx_2_id = inv_spec
        # Grab only obs id, species id, genus, family because lat /lon not necessary at the moment
        self.num_specs = len(obs.species_id.unique())
        self.num_fams = len(obs.family.unique())
        self.num_gens = len(obs.genus.unique())
        self.spec_freqs = obs.species_id.value_counts().to_dict()
        self.gen_freqs = obs.genus.value_counts().to_dict()
        self.fam_freqs = obs.family.value_counts().to_dict()                
        # convert to numpy
        obs = obs[:500]        
        self.obs = obs[['id', 'species_id', 'genus', 'family']].values
        self.transform = transform
        channels, alt_shape, rgbd_shape = get_shapes(self.obs[0,0], self.base_dir, self.altitude)
        self.channels = channels
        self.alt_shape = alt_shape
        self.rgbd_shape = rgbd_shape

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # obs is of shape [id, species_id, genus, family]    
        id_ = self.obs[idx, 0]
        images = image_from_id(id_, self.base_dir, self.altitude)
        composite_label = self.obs[idx, 1:] # get genus, family as well
        if self.transform:
            images = self.transform(images)
        return (composite_label, images)
    
    
class GEOCELF_Dataset_Joint_Scalar_Raster(Dataset):
    def __init__(self, base_dir, organism, altitude, country='us', transform=None, normalize=True):
        self.base_dir = base_dir
        self.country = country
        self.organism = organism
        obs = get_joint_gbif_data(self.base_dir, country, organism)
        rasterpath = "{}rasters".format(self.base_dir)
        rasters  = PatchExtractor(rasterpath, size = 1)
        rasters.add_all(normalized=normalize)
        self.rasters = rasters
        self.altitude = altitude
        obs.fillna('nan', inplace=True)        
        obs, inv_spec = prep_joint_data(obs)
        self.idx_2_id = inv_spec
        # Grab only obs id, species id, genus, family because lat /lon not necessary at the moment
        self.num_specs = len(obs.species_id.unique())
        self.num_fams = len(obs.family.unique())
        self.num_gens = len(obs.genus.unique())
        self.spec_freqs = obs.species_id.value_counts().to_dict()
        self.gen_freqs = obs.genus.value_counts().to_dict()
        self.fam_freqs = obs.family.value_counts().to_dict()                
        self.obs = obs[['id', 'all_specs', 'all_fams', 'all_gens', 'lat_lon']].values
        self.transform = transform
        channels, alt_shape, rgbd_shape = get_shapes(self.obs[0,0], self.base_dir, self.altitude)
        self.channels = channels
        self.num_rasters = len(rasters) 
        self.alt_shape = alt_shape
        self.rgbd_shape = rgbd_shape
    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get images
        # obs is of shape [id, species_id, genus, family]    
        id_ = self.obs[idx, 0]
        images = image_from_id(id_, self.base_dir, self.altitude)     
        # get raster data
        lat_lon = self.obs[idx, 4]
        env_rasters = self.rasters[lat_lon]
        # get labels
        specs_label = self.obs[idx, 1]
        gens_label = self.obs[idx, 3]
        fams_label = self.obs[idx, 2]        

        if self.transform:
            images = self.transform(images)
        return (specs_label, gens_label, fams_label, images, env_rasters)        


    
class GEOCELF_Dataset_Joint_Scalar_Raster_LatLon(Dataset):
    def __init__(self, base_dir, organism, country='us', transform=None, normalize=True):
        self.base_dir = base_dir
        self.country = country
        self.organism = organism
        obs = get_joint_gbif_data(self.base_dir, country, organism)
        rasterpath = "{}rasters".format(self.base_dir)
        rasters  = PatchExtractor(rasterpath, size = 1)
        rasters.add_all(normalized=normalize)
        self.rasters = rasters
        obs.fillna('nan', inplace=True)        
        obs, inv_spec = prep_joint_data(obs)
        self.idx_2_id = inv_spec
        # Grab only obs id, species id, genus, family because lat /lon not necessary at the moment
        self.num_specs = len(obs.species_id.unique())
        self.num_fams = len(obs.family.unique())
        self.num_gens = len(obs.genus.unique())
        self.spec_freqs = obs.species_id.value_counts().to_dict()
        self.gen_freqs = obs.genus.value_counts().to_dict()
        self.fam_freqs = obs.family.value_counts().to_dict()    
        self.altitude = altitude        
        self.lat_scale = obs.lat.max()-obs.lat.min()
        self.lon_scale = obs.lon.max()-obs.lon.min()
        self.lat_min = obs.lat.min()
        self.lon_min = obs.lon.min()        
        self.normalize = normalize
        
        self.obs = obs[['id', 'all_specs', 'all_fams', 'all_gens', 'lat_lon', 'lat', 'lon']].values
        self.transform = transform
        channels, alt_shape, rgbd_shape = get_shapes(self.obs[0,0], self.base_dir, self.altitude)
        self.channels = channels
        self.num_rasters = len(rasters)+ 2 # plus two because including the lat lon
        self.alt_shape = alt_shape
        self.rgbd_shape = rgbd_shape
    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get images
        # obs is of shape [id, species_id, genus, family]    
        id_ = self.obs[idx, 0]
        images = image_from_id(id_, self.base_dir, self.altitude)     
        # get raster data
        lat_lon = self.obs[idx, 4]
        env_rasters = self.rasters[lat_lon]
        if self.normalize:
            
            env_rasters.append(utils.normalize(self.obs[idx, 5], self.lat_min, self.lat_scale))
            env_rasters.append(utils.normalize(self.obs[idx, 6], self.lon_min, self.lon_scale))
                
        else:
            # add lat lon data unnormalized
            env_rasters.append(self.obs[idx, 5])
            env_rasters.append(self.obs[idx, 6])            
        # get labels
        specs_label = self.obs[idx, 1]
        gens_label = self.obs[idx, 3]
        fams_label = self.obs[idx, 2]        

        if self.transform:
            images = self.transform(images)
        return (specs_label, gens_label, fams_label, images, env_rasters)        




    # x, y = eniffa * (get_item_from_obs(obs,1)[1], get_item_from_obs(obs,1)[0])
class GEOCELF_Dataset_BioClim_Only(Dataset):
    def __init__(self, base_dir, organism, country='cali', transform=None, normalize='none'):
        self.base_dir = base_dir
        self.country = country
        self.organism = organism
        self.channels = None
        obs = get_gbif_rasters_data(self.base_dir, country, organism)
        rasterpath = "{}rasters".format(self.base_dir)
        self.rasters, self.affine, obs, self.nan = get_bioclim_rasters(base_dir, country, normalize, obs)
        obs.fillna('nan', inplace=True)        
        obs, inv_spec = prep_joint_data(obs)
        self.idx_2_id = inv_spec
        # Grab only obs id, species id, genus, family because lat /lon not necessary at the moment
        self.num_specs = len(obs.species_id.unique())
        self.num_fams = len(obs.family.unique())
        self.num_gens = len(obs.genus.unique())
        self.spec_freqs = obs.species_id.value_counts().to_dict()
        self.gen_freqs = obs.genus.value_counts().to_dict()
        self.fam_freqs = obs.family.value_counts().to_dict()        
        self.lat_scale = obs.lat.max()-obs.lat.min()
        self.lon_scale = obs.lon.max()-obs.lon.min()
        self.lat_min = obs.lat.min()
        self.lon_min = obs.lon.min()        
        self.normalize = normalize
        self.obs = obs[['id', 'all_specs', 'all_fams', 'all_gens', 'lat_lon', 'lat', 'lon']].values
        self.num_rasters = self.rasters.shape[0]+ 2 # plus two because including the lat lon
        print("num rasters is ", self.num_rasters)

    def __len__(self):
        return len(self.obs)
    # assumes the latlon format from gbif observation building
    def latlon_2_index(self, latlon):
        y, x =  ~self.affine * (latlon[1], latlon[0])
        return int(round(x)), int(round(y))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get raster data
        lat_lon = self.obs[idx, 4]
        x, y = self.latlon_2_index(lat_lon)
        env_rasters = self.rasters[:,x,y]
        assert (env_rasters == nan).sum() == 0, "attempting to index an observation outside the coordinate range at {} for obs index {} value is {} and nan is {}".format(lat_lon, id_, env_rasters, nan)
        
        if self.normalize:
            lat_norm = utils.normalize_latlon(self.obs[idx, 5], self.lat_min, self.lat_scale)
            lon_norm = utils.normalize_latlon(self.obs[idx, 6], self.lon_min, self.lon_scale)
            env_rasters = np.append(env_rasters, [lat_norm, lon_norm])
                
        else:
            # add lat lon data unnormalized
            env_rasters = np.append(env_rasters, [self.obs[idx, 5], self.obs[idx, 6]])
        # get labels
        assert len(env_rasters) == self.num_rasters, "raster sizes don't match"
        specs_label = self.obs[idx, 1]
        gens_label = self.obs[idx, 3]
        fams_label = self.obs[idx, 2]        
        return (specs_label, gens_label, fams_label, env_rasters)

        # x, y = eniffa * (get_item_from_obs(obs,1)[1], get_item_from_obs(obs,1)[0])
class GEOCELF_Dataset_BioClim_CNN(Dataset):
    def __init__(self, base_dir, organism, country='cali', transform=None, normalize='none', pix_res=256, big=True):
        self.base_dir = base_dir
        self.country = country
        self.organism = organism
        self.pix_res = pix_res
        obs = get_gbif_rasters_data(self.base_dir, country, organism)
        rasterpath = "{}rasters".format(self.base_dir)
        self.rasters, self.affine, obs, self.nan = get_bioclim_rasters(base_dir, country, normalize, obs, big)
        obs.fillna('nan', inplace=True)        
        obs, inv_spec = prep_joint_data(obs)
        self.idx_2_id = inv_spec
        # Grab only obs id, species id, genus, family because lat /lon not necessary at the moment
        self.num_specs = len(obs.species_id.unique())
        self.num_fams = len(obs.family.unique())
        self.num_gens = len(obs.genus.unique())
        self.spec_freqs = obs.species_id.value_counts().to_dict()
        self.gen_freqs = obs.genus.value_counts().to_dict()
        self.fam_freqs = obs.family.value_counts().to_dict()        
        self.lat_scale = obs.lat.max()-obs.lat.min()
        self.lon_scale = obs.lon.max()-obs.lon.min()
        self.lat_min = obs.lat.min()
        self.lon_min = obs.lon.min()        
        self.normalize = normalize
        self.obs = obs[['id', 'all_specs', 'all_fams', 'all_gens', 'lat_lon', 'lat', 'lon']].values
        self.channels = self.rasters.shape[0]
        print("num rasters is ", self.channels)

    def __len__(self):
        return len(self.obs)
    # assumes the latlon format from gbif observation building
    def latlon_2_index(self, latlon):
        y, x =  ~self.affine * (latlon[1], latlon[0])
        return int(round(x)), int(round(y))
    
    


    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get raster data
        lat_lon = self.obs[idx, 4]
        x, y = self.latlon_2_index(lat_lon)
        xmin, xmax, ymin, ymax = xy_2_range_center(self.pix_res, x, y)
        # the bioclim rasters don't extend a full 100 km off the western coast of cali, so need to impute nan for westernmost
        # observations to account for this fact and still be able to use observations for these points
        
        env_rasters = raster_cnn_image(self.rasters, xmin, xmax, ymin, ymax, self.nan)
        # get labels
        specs_label = self.obs[idx, 1]
        gens_label = self.obs[idx, 3]
        fams_label = self.obs[idx, 2]
        return (specs_label, gens_label, fams_label, env_rasters)

class GEOCELF_Dataset_Joint_BioClim_LowRes(Dataset):
    def __init__(self, base_dir, organism, altitude, normalize, country='us', transform=None):
        self.base_dir = base_dir
        self.country = country
        self.organism = organism
        obs = get_gbif_rasters_data(self.base_dir, country, organism)
        rasterpath = "{}rasters".format(self.base_dir)
        self.rasters, self.affine, obs, self.nan = get_bioclim_rasters(base_dir, country, normalize, obs, True)
        self.altitude = altitude
        obs.fillna('nan', inplace=True)        
        obs, inv_spec = prep_joint_data(obs)
        self.idx_2_id = inv_spec
        # Grab only obs id, species id, genus, family because lat /lon not necessary at the moment
        self.num_specs = len(obs.species_id.unique())
        self.num_fams = len(obs.family.unique())
        self.num_gens = len(obs.genus.unique())
        self.spec_freqs = obs.species_id.value_counts().to_dict()
        self.gen_freqs = obs.genus.value_counts().to_dict()
        self.fam_freqs = obs.family.value_counts().to_dict()                
        self.obs = obs[['id', 'all_specs', 'all_fams', 'all_gens', 'lat_lon', 'lat', 'lon']].values
        self.normalize = normalize
        self.transform = transform
        self.lat_scale = obs.lat.max()-obs.lat.min()
        self.lon_scale = obs.lon.max()-obs.lon.min()
        self.lat_min = obs.lat.min()
        self.lon_min = obs.lon.min()
        channels, width, height = get_shapes(self.obs[0,0], self.base_dir, self.altitude)
        self.pix_res = width
        assert width == height, "the width and height of input images dont match!"
        self.num_rasters = len(self.rasters) 
        self.channels = channels + self.num_rasters
        self.width = width
        self.height = height
 
    def __len__(self):
        return len(self.obs)
    
    def latlon_2_index(self, latlon):
        y, x =  ~self.affine * (latlon[1], latlon[0])
        return int(round(x)), int(round(y))

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get images
        # obs is of shape [id, species_id, genus, family]    
        id_ = self.obs[idx, 0]
        images = image_from_id(id_, self.base_dir, self.altitude)     
        if self.transform:
            images = self.transform(images)
        # get raster data
        lat_lon = self.obs[idx, 4]
        x, y = self.latlon_2_index(lat_lon)
        xmin, xmax, ymin, ymax = xy_2_range_center(self.pix_res, x, y)
        # the bioclim rasters don't extend a full 100 km off the western coast of cali, so need to impute nan for westernmost
        # observations to account for this fact and still be able to use observations for these points
        
        env_rasters = raster_cnn_image(self.rasters, xmin, xmax, ymin, ymax, self.nan)
        # get labels
        assert len(env_rasters) == self.num_rasters, "raster sizes don't match"
#         import pdb; pdb.set_trace()
        all_imgs = np.concatenate([images, env_rasters], axis=0)
        specs_label = self.obs[idx, 1]
        gens_label = self.obs[idx, 3]
        fams_label = self.obs[idx, 2]        
        return (specs_label, gens_label, fams_label, all_imgs)    
    
class GEOCELF_Dataset_Joint_BioClim(Dataset):
        
    def __init__(self, base_dir, organism, country='cali', altitude=True, transform=None, normalize='none'):
        self.base_dir = base_dir
        self.country = country
        self.organism = organism
        obs = get_gbif_rasters_data(self.base_dir, country, organism)
        rasterpath = "{}rasters".format(self.base_dir)
        self.rasters, self.affine, obs, self.nan = get_bioclim_rasters(base_dir, country, normalize, obs)
        obs.fillna('nan', inplace=True)        
        obs, inv_spec = prep_joint_data(obs)
        self.idx_2_id = inv_spec
        # Grab only obs id, species id, genus, family because lat /lon not necessary at the moment
        self.num_specs = len(obs.species_id.unique())
        self.num_fams = len(obs.family.unique())
        self.num_gens = len(obs.genus.unique())
        self.spec_freqs = obs.species_id.value_counts().to_dict()
        self.gen_freqs = obs.genus.value_counts().to_dict()
        self.fam_freqs = obs.family.value_counts().to_dict()        
        self.lat_scale = obs.lat.max()-obs.lat.min()
        self.lon_scale = obs.lon.max()-obs.lon.min()
        self.lat_min = obs.lat.min()
        self.lon_min = obs.lon.min()        
        self.normalize = normalize
        self.altitude = altitude
        self.obs = obs[['id', 'all_specs', 'all_fams', 'all_gens', 'lat_lon', 'lat', 'lon']].values
        self.transform = transform
        channels, alt_shape, rgbd_shape = get_shapes(self.obs[0,0], self.base_dir, self.altitude)
        self.channels = channels

        self.num_rasters = self.rasters.shape[0]+ 2 # plus two because including the lat lon
        print("num rasters is ", self.num_rasters)
        self.alt_shape = alt_shape
        self.rgbd_shape = rgbd_shape
    def __len__(self):
        return len(self.obs)
    # assumes the latlon format from gbif observation building
    def latlon_2_index(self, latlon):
        y, x =  ~self.affine * (latlon[1], latlon[0])
        return int(round(x)), int(round(y))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get images
        # obs is of shape [id, species_id, genus, family]    
        id_ = self.obs[idx, 0]
        images = image_from_id(id_, self.base_dir, self.altitude)     
        # get raster data
        lat_lon = self.obs[idx, 4]
        x, y = self.latlon_2_index(lat_lon)
        env_rasters = self.rasters[:,x,y]
        assert (env_rasters == nan).sum() == 0, "attempting to index an observation outside the coordinate range at {} for obs index {} value is {} and nan is {}".format(lat_lon, id_, env_rasters, nan)
        
        if self.normalize:
            lat_norm = utils.normalize_latlon(self.obs[idx, 5], self.lat_min, self.lat_scale)
            lon_norm = utils.normalize_latlon(self.obs[idx, 6], self.lon_min, self.lon_scale)
            env_rasters = np.append(env_rasters, [lat_norm, lon_norm])
                
        else:
            # add lat lon data unnormalized
            env_rasters = np.append(env_rasters, [self.obs[idx, 5], self.obs[idx, 6]])
        # get labels
        assert len(env_rasters) == self.num_rasters, "raster sizes don't match"
        specs_label = self.obs[idx, 1]
        gens_label = self.obs[idx, 3]
        fams_label = self.obs[idx, 2]        
        if self.transform:
            images = self.transform(images)
        return (specs_label, gens_label, fams_label, images, env_rasters)
    
class GEOCELF_Dataset_Joint_BioClim_Sheet(Dataset):
    def __init__(self, base_dir, organism, altitude, normalize, country='us', transform=None):
        self.base_dir = base_dir
        self.country = country
        self.organism = organism
        obs = get_gbif_rasters_data(self.base_dir, country, organism)
        rasterpath = "{}rasters".format(self.base_dir)
        self.rasters, self.affine, obs, self.nan = get_bioclim_rasters(base_dir, country, normalize, obs)
        self.altitude = altitude
        obs.fillna('nan', inplace=True)        
        obs, inv_spec = prep_joint_data(obs)
        self.idx_2_id = inv_spec
        # Grab only obs id, species id, genus, family because lat /lon not necessary at the moment
        self.num_specs = len(obs.species_id.unique())
        self.num_fams = len(obs.family.unique())
        self.num_gens = len(obs.genus.unique())
        self.spec_freqs = obs.species_id.value_counts().to_dict()
        self.gen_freqs = obs.genus.value_counts().to_dict()
        self.fam_freqs = obs.family.value_counts().to_dict()                
        self.obs = obs[['id', 'all_specs', 'all_fams', 'all_gens', 'lat_lon', 'lat', 'lon']].values
        self.normalize = normalize
        self.transform = transform
        self.lat_scale = obs.lat.max()-obs.lat.min()
        self.lon_scale = obs.lon.max()-obs.lon.min()
        self.lat_min = obs.lat.min()
        self.lon_min = obs.lon.min()
        channels, width, height = get_shapes(self.obs[0,0], self.base_dir, self.altitude)
        self.num_rasters = len(self.rasters) + 2
        self.channels = channels + self.num_rasters
        self.width = width
        self.height = height
 
    def __len__(self):
        return len(self.obs)
    
    def latlon_2_index(self, latlon):
        y, x =  ~self.affine * (latlon[1], latlon[0])
        return int(round(x)), int(round(y))

    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get images
        # obs is of shape [id, species_id, genus, family]    
        id_ = self.obs[idx, 0]
        images = image_from_id(id_, self.base_dir, self.altitude)     
        if self.transform:
            images = self.transform(images)
        # get raster data
        lat_lon = self.obs[idx, 4]
        x, y = self.latlon_2_index(lat_lon)
        env_rasters = self.rasters[:,x,y]
        assert (env_rasters == nan).sum() == 0, "attempting to index an observation outside the coordinate range at {} for obs index {} value is {} and nan is {}".format(lat_lon, id_, env_rasters, nan)
        
        if self.normalize == 'min_max':
            lat_norm = utils.normalize_latlon(self.obs[idx, 5], self.lat_min, self.lat_scale)
            lon_norm = utils.normalize_latlon(self.obs[idx, 6], self.lon_min, self.lon_scale)
            env_rasters = np.append(env_rasters, [lat_norm, lon_norm])
                
        elif self.normalize == 'normalize':
            raise NotImplementedError
        else:
            # add lat lon data unnormalized
            env_rasters = np.append(env_rasters, [self.obs[idx, 5], self.obs[idx, 6]])
        # get labels
        assert len(env_rasters) == self.num_rasters, "raster sizes don't match"
        ras_cnn = [np.full((self.width, self.height),  val) for val in env_rasters]
        ras_cnn = np.stack(ras_cnn)
        all_imgs = np.concatenate([images, ras_cnn], axis=0)
        specs_label = self.obs[idx, 1]
        gens_label = self.obs[idx, 3]
        fams_label = self.obs[idx, 2]        
        return (specs_label, gens_label, fams_label, all_imgs)