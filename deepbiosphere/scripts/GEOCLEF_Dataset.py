from collections import Counter
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


def get_gbif_data(pth, split, region, organism):
    ## Grab GBIF observation data

    obs_pth = "{}occurrences/occurrences_{}_{}_{}.csv".format(pth, region, organism, split)
    return pd.read_csv(obs_pth, sep=';')  

def get_gbif_rasters_data(pth, region, organism):
#     {pth}/occurrences/joint_obs_{region}{plant}_train_rasters.csv
    obs_pth = "{}occurrences/joint_obs_{}_{}_train_rasters.csv".format(pth, region, organism)
    joint_obs = pd.read_csv(obs_pth)  
    joint_obs.all_specs = joint_obs.all_specs.apply(lambda x: parse_string_to_string(x))
    joint_obs.all_gens = joint_obs.all_gens.apply(lambda x: parse_string_to_string(x))
    joint_obs.all_fams = joint_obs.all_fams.apply(lambda x: parse_string_to_string(x))
    joint_obs.lat_lon = joint_obs.lat_lon.apply(lambda x: parse_string_to_tuple(x))
    return joint_obs

def get_joint_gbif_data(pth, region, organism):
    ## Grab GBIF observation data
    obs_pth = "{}occurrences/joint_multiple_obs_{}_{}.csv".format(pth, region, organism)
    joint_obs = pd.read_csv(obs_pth)  
    joint_obs.all_specs = joint_obs.all_specs.apply(lambda x: parse_string_to_string(x))
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
def latlon_2_idx(affine, latlon):
    y, x =  ~affine * (latlon[1], latlon[0])
    return int(round(x)), int(round(y))

        # the bioclim rasters don't extend a full 100 km off the western coast of cali, so need to impute nan for westernmost
        # observations to account for this fact and still be able to use observations for these points
def get_raster_image_obs(lat_lon, affine, rasters, nan, normalize, pix_res):
    x, y = latlon_2_idx(affine, lat_lon)
    xmin, xmax, ymin, ymax = xy_2_range_center(pix_res, x, y)    
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
        print("riperoni out of bounds")
        print(xmin, xmax, ymin, ymax, rasters.shape)
        exit(1), "{} is outside bounds of env raster image!".format(lat_lon)
    else: 
        env_rasters = rasters[:,xmin:xmax,ymin:ymax]
    return env_rasters
def get_raster_point_obs(lat_lon, affine, rasters, nan, normalize, lat_min, lat_max, lon_min, lon_max):
    x, y = latlon_2_idx(affine, lat_lon)
    env_rasters = rasters[:,x,y]
    if normalize == 'min_max':
        lat_norm = utils.scale(lat_lon[0], min_= lat_min, max_= lat_max)
        lon_norm = utils.scale(lat_lon[1], min_= lon_min, max_= lon_max)
        env_rasters = np.append(env_rasters, [lat_norm, lon_norm])

    elif  normalize == 'normalize':
        raise NotImplementedError
    else:
        env_rasters = np.append(env_rasters, [lat_lon[0], lat_lon[1]])    
    return env_rasters

def get_raster_sheet_obs(lat_lon, affine, rasters, nan, normalize, lat_min, lat_max, lon_min, lon_max, width, height):
    x, y = latlon_2_idx(affine, lat_lon)
    env_rasters = rasters[:,x,y]
    if normalize == 'min_max':
        lat_norm = utils.scale(lat_lon[0], min_= lat_min, max_= lat_max)
        lon_norm = utils.scale(lat_lon[1], min_= lon_min, max_= lon_max)
        env_rasters = np.append(env_rasters, [lat_norm, lon_norm])

    elif  normalize == 'normalize':
        raise NotImplementedError
    else:
        env_rasters = np.append(env_rasters, [lat_lon[0], lat_lon[1]])    
    ras_cnn = [np.full((width, height),  val) for val in env_rasters]
    ras_cnn = np.stack(ras_cnn)
    return ras_cnn
def raster_filter_2_cali(base_dir, obs):
    
    geoms = get_cali_shape(base_dir)
    ras_paths = get_us_bioclim(base_dir)
    src = open_raster(ras_paths[0])       
# filter down the dataset to only in-range observations
    filt_obs = filter_to_bioclim(obs, src, geoms, nan)
    return  filt_obs

def get_bioclim_rasters(base_dir, region, normalized, obs):
    
    if region ==  'cali':
        # grab the raster of cali shape with buffer
        # has a ~100 km radius around the cali border for any observation that sits right on the edge
        geoms = get_big_cali_shape(base_dir) # old:  get_cali_shape(base_dir) if not big else 
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
    # this shape of cali is the us-census designated shape
#     geoms = get_cali_shape(base_dir)
#     filt_obs = filter_to_bioclim(obs, src, geoms, nan)
    # make sure rasters in same affine 
    # replace nan value with something more reasonable
    # nan is 2x more negative than most smallest value across all rasters
    mins = [r.min() for r in ras_agg]
    maxs = [r.max() for r in ras_agg]
    min_ = max(maxs) - min(mins) * 2
    min_ = -min_
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
    
    return all_ras, affine, obs, min_

def filter_to_bioclim(obs, src, geoms, nan):
    bad_ids = []
    masked, affine = mask(src, geoms, nodata=nan, filled=True, crop=False, pad=True, all_touched=True)
    masked = np.squeeze(masked)
    for i, ob in obs.iterrows():
        x, y = latlon_2_idx(affine, ob.lat_lon)
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
    # channels, width, height
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


def map_key_2_index(df, key, new_key=None):
    key_2_id = {
        k:v for k, v in 
        zip(df[key].unique(), np.arange(len(df[key].unique())))
    }
    if new_key == None:
        df[key] = df[key].map(key_2_id)
    else:
        df[new_key] = df[key].map(key_2_id)
    return df, key_2_id
# this is non-deterministic, fuck
def map_unq_2_index(df, key):
    all_itm = df[key]
    unq = set()
    for itm in all_itm:
        unq.update(itm)
    name_2_id = {
        k:v for k, v in zip(unq, np.arange(len(unq)))
    }
    return name_2_id

def dict_key_2_index(df, key):
    return {
        k:v for k, v in 
        zip(df[key].unique(), np.arange(len(df[key].unique())))
    }


# def prep_data(obs):

#     obs, spec_dict = map_key_2_index(obs, 'species_id')
#     inv_spec = {v: k for k, v in spec_dict.items()}
#     obs, _ = map_key_2_index(obs, 'genus', 'genus_id')
#     obs, _ = map_key_2_index(obs, 'family', 'family_id')

#     return obs, inv_spec    
    
# TODO: assumes that species_id, genus, family columns contain all possible values contained in extra_obs    :398
def prep_data(obs, observation):

    # map all species ids to 0-num_species, same for family and genus
    obs, spec_dict = map_key_2_index(obs, 'species', 'species_id')
    inv_spec = {v: k for k, v in spec_dict.items()}
    obs, gen_dict = map_key_2_index(obs, 'genus', 'genus_id')
    obs, fam_dict = map_key_2_index(obs, 'family', 'family_id')
    if observation == 'joint_single' or observation == 'single_single':
        spec_dict = map_unq_2_index(obs, 'all_specs')
        inv_spec = {v: k for k, v in spec_dict.items()}        
        fam_dict = map_unq_2_index(obs, 'all_fams')
        gen_dict = map_unq_2_index(obs, 'all_gens')
        obs = obs.assign(species_id=[spec_dict[k] for k in obs.species])
        obs = obs.assign(genus_id=[gen_dict[k] for k in obs.genus])
        obs = obs.assign(family_id=[fam_dict[k] for k in obs.family])        
    # also map all species / genus / family in joint observation to 0-num
    obs = obs.assign(all_specs=[[spec_dict[k] for k in row ] for row in obs.all_specs])
    obs = obs.assign(all_gens=[[gen_dict[k] for k in row ] for row in obs.all_gens])
    obs = obs.assign(all_fams=[[fam_dict[k] for k in row ] for row in obs.all_fams])
        
    return obs, inv_spec, spec_dict, gen_dict, fam_dict

def get_labels(observation, obs, idx):
        if observation == 'single' or observation == 'single_single':
            specs_label = obs[idx, sp_idx]
            gens_label = obs[idx, gen_idx]
            fams_label = obs[idx, fam_idx]            
        else:
            specs_label = obs[idx, all_sp_idx]
            gens_label = obs[idx,  all_gen_idx]
            fams_label = obs[idx,  all_fam_idx]            
        return specs_label, gens_label, fams_label


def get_inference_labels(observation, obs, idx):
        specs_label = obs[idx, sp_idx]
        gens_label = obs[idx, gen_idx]
        fams_label = obs[idx, fam_idx]            
        all_spec = obs[idx, all_sp_idx]
        all_gen = obs[idx,  all_gen_idx]
        all_fam = obs[idx,  all_fam_idx]            
        return specs_label, gens_label, fams_label, all_spec, all_gen, all_fam
    
    
def get_gbif_observations(base_dir, organism, region, observation):
    #TODO: grab the right gbif dataset depending on what region, what observation type, what organism
    # even for single observation, go ahead and grab the joint dataset, will just choose to not use joint data later on when grabbing observation
    # include get_gbif_rasters_data!!
    if observation == 'single':
        observation = 'joint_multiple'

    elif observation == 'single_single':
        observation = 'joint_single'
    obs_pth = "{}occurrences/{}_obs_{}_{}_train.csv".format(base_dir, observation, region, organism)
    joint_obs = pd.read_csv(obs_pth, sep=',')
    joint_obs.all_specs = joint_obs.all_specs.apply(lambda x: parse_string_to_string(x))
    joint_obs.all_gens = joint_obs.all_gens.apply(lambda x: parse_string_to_string(x))
    joint_obs.all_fams = joint_obs.all_fams.apply(lambda x: parse_string_to_string(x))
    joint_obs.lat_lon = joint_obs.lat_lon.apply(lambda x: parse_string_to_tuple(x))
    return joint_obs

    return joint_obs
id_idx = 0
sp_idx = 1
gen_idx = 2
fam_idx = 3
all_sp_idx = 4
all_gen_idx = 6
all_fam_idx = 5
lat_lon_idx = 7




# just the high resolution satellite imagery
class HighRes_Satellie_Images_Only(Dataset):
    def __init__(self, base_dir, organism, region, observation, altitude):
        self.base_dir = base_dir
        self.region = region
        self.organism = organism
        self.altitude = altitude
        self.observation = observation
        obs = get_gbif_observations(base_dir,organism, region, observation)
        obs.fillna('nan', inplace=True)
        if 'species' not in obs.columns:
            obs = utils.add_taxon_metadata(self.base_dir, obs, self.organism)
        
        obs, inv_spec, spec_dict, gen_dict, fam_dict  = prep_data(obs, observation)
        self.idx_2_id = inv_spec
        # Grab only obs id, species id, genus, family because lat /lon not necessary at the moment
        self.num_specs = len(spec_dict)
        self.num_fams = len(fam_dict)
        self.num_gens = len(gen_dict)
        if observation == 'joint_single' or observation == 'single_single':
            all_sps = [sp for ob in obs.all_specs for sp in ob]
            all_gen = [sp for ob in obs.all_gens for sp in ob]
            all_fam = [sp for ob in obs.all_fams for sp in ob]
            self.spec_freqs =Counter(all_sps) 
            self.gen_freqs = Counter(all_gen)
            self.fam_freqs = Counter(all_fam)

        else:
            self.spec_freqs = obs.species_id.value_counts().to_dict()
            self.gen_freqs = obs.genus_id.value_counts().to_dict()
            self.fam_freqs = obs.family_id.value_counts().to_dict()                
        # convert to numpy
        self.obs = obs[['id', 'species_id', 'genus_id', 'family_id', 'all_specs', 'all_fams', 'all_gens']].values

        channels, width, height = get_shapes(self.obs[0,0], self.base_dir, self.altitude)
        self.channels = channels
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # obs is of shape [id, species_id, genus, family]    
        id_ = self.obs[idx, id_idx]
        images = image_from_id(id_, self.base_dir, self.altitude)

        specs_label, gens_label, fams_label = get_labels(self.observation, self.obs, idx)
        return (specs_label, gens_label, fams_label, images)
    
    
    def infer_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # obs is of shape [id, species_id, genus, family]    
        id_ = self.obs[idx, id_idx]
        images = image_from_id(id_, self.base_dir, self.altitude)

        specs_label, gens_label, fams_label, all_spec, all_gen, all_fam = get_inference_labels(self.observation, self.obs, idx)
        return (specs_label, gens_label, fams_label, all_spec, all_gen, all_fam, images)

    # the high resolution satellite imagery + the pointwise observation environmental rasters
class HighRes_Satellite_Rasters_Point(Dataset):
    def __init__(self, base_dir, organism, region, observation, altitude, normalize):
        self.base_dir = base_dir
        self.region = region
        self.organism = organism
        self.altitude = altitude
        self.normalize = normalize
        self.observation = observation
        obs = get_gbif_observations(base_dir,organism, region, observation)
        rasterpath = "{}rasters".format(self.base_dir)
        self.rasters, self.affine, obs, self.nan = get_bioclim_rasters(base_dir, region, normalize, obs)
        obs.fillna('nan', inplace=True)               
        if 'species' not in obs.columns:
            obs = utils.add_taxon_metadata(self.base_dir, obs, self.organism)

        obs, inv_spec, spec_dict, gen_dict, fam_dict  = prep_data(obs, observation)
        self.idx_2_id = inv_spec
        # Grab only obs id, species id, genus, family because lat /lon not necessary at the moment
        self.num_specs = len(spec_dict)
        self.num_fams = len(fam_dict)
        self.num_gens = len(gen_dict)
        if observation == 'joint_single'  or observation == 'single_single':
            all_sps = [sp for ob in obs.all_specs for sp in ob]
            all_gen = [sp for ob in obs.all_gens for sp in ob]
            all_fam = [sp for ob in obs.all_fams for sp in ob]
            self.spec_freqs =Counter(all_sps) 
            self.gen_freqs = Counter(all_gen)
            self.fam_freqs = Counter(all_fam)

        else:
            self.spec_freqs = obs.species_id.value_counts().to_dict()
            self.gen_freqs = obs.genus_id.value_counts().to_dict()
            self.fam_freqs = obs.family_id.value_counts().to_dict()
        self.lat_max = obs.lat.max()
        self.lon_max = obs.lon.max()
        self.lat_min = obs.lat.min()
        self.lon_min = obs.lon.min()        
        self.num_rasters = self.rasters.shape[0]+ 2 # plus two because including the lat lon
        print("num rasters is ", self.num_rasters)

        # convert to numpy
        self.obs = obs[['id', 'species_id', 'genus_id', 'family_id', 'all_specs', 'all_fams', 'all_gens', 'lat_lon']].values

        channels, width, height = get_shapes(self.obs[0,0], self.base_dir, self.altitude)
        self.channels = channels
        self.width = width
        self.height = height
    def __len__(self):
        return len(self.obs)
    # assumes the latlon format from gbif observation building
    def latlon_2_idx(self, latlon):
        y, x =  ~self.affine * (latlon[1], latlon[0])
        return int(round(x)), int(round(y))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get images  
        id_ = self.obs[idx, id_idx]
        images = image_from_id(id_, self.base_dir, self.altitude)
        # get raster data
        lat_lon = self.obs[idx, lat_lon_idx]
        env_rasters = get_raster_point_obs(lat_lon, self.affine, self.rasters, self.nan, self.normalize, self.lat_min, self.lat_max, self.lon_min, self.lon_max)
        # get labels
        specs_label, gens_label, fams_label = get_labels(self.observation, self.obs, idx)
        return (specs_label, gens_label, fams_label, images, env_rasters)

    def infer_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get images  
        id_ = self.obs[idx, id_idx]
        images = image_from_id(id_, self.base_dir, self.altitude)
        # get raster data
        lat_lon = self.obs[idx, lat_lon_idx]
        env_rasters = get_raster_point_obs(lat_lon, self.affine, self.rasters, self.nan, self.normalize, self.lat_min, self.lat_max, self.lon_min, self.lon_max)
        # get labels
        specs_label, gens_label, fams_label, all_spec, all_gen, all_fam = get_inference_labels(self.observation, self.obs, idx)
        return (specs_label, gens_label, fams_label, all_spec, all_gen, all_fam, images, env_rasters)    
    
    
    # x, y = eniffa * (get_item_from_obs(obs,1)[1], 
    # just the environmental raster point value at a location
class Bioclim_Rasters_Point(Dataset):
    def __init__(self, base_dir, organism, region, normalize, observation):
        self.base_dir = base_dir
        self.region = region
        self.organism = organism
        self.channels = None
        self.normalize = normalize
        self.observation = observation
        obs = get_gbif_observations(base_dir,organism, region, observation)
        rasterpath = "{}rasters".format(self.base_dir)
        self.rasters, self.affine, obs, self.nan = get_bioclim_rasters(base_dir, region, normalize, obs)
        obs.fillna('nan', inplace=True)               
        if 'species' not in obs.columns:
            obs = utils.add_taxon_metadata(self.base_dir, obs, self.organism)

        obs, inv_spec, spec_dict, gen_dict, fam_dict  = prep_data(obs, observation)
        self.idx_2_id = inv_spec
        # Grab only obs id, species id, genus, family because lat /lon not necessary at the moment
        self.num_specs = len(spec_dict)
        self.num_fams = len(fam_dict)
        self.num_gens = len(gen_dict)
        if observation == 'joint_single'  or observation == 'single_single':
            all_sps = [sp for ob in obs.all_specs for sp in ob]
            all_gen = [sp for ob in obs.all_gens for sp in ob]
            all_fam = [sp for ob in obs.all_fams for sp in ob]
            self.spec_freqs =Counter(all_sps) 
            self.gen_freqs = Counter(all_gen)
            self.fam_freqs = Counter(all_fam)

        else:
            self.spec_freqs = obs.species_id.value_counts().to_dict()
            self.gen_freqs = obs.genus_id.value_counts().to_dict()
            self.fam_freqs = obs.family_id.value_counts().to_dict()
        self.lat_max = obs.lat.max()
        self.lon_max = obs.lon.max()
        self.lat_min = obs.lat.min()
        self.lon_min = obs.lon.min()        
        self.num_rasters = self.rasters.shape[0]+ 2 # plus two because including the lat lon
        print("num rasters is ", self.num_rasters)

        # convert to numpy
        self.obs = obs[['id', 'species_id', 'genus_id', 'family_id', 'all_specs', 'all_fams', 'all_gens', 'lat_lon']].values

    def __len__(self):
        return len(self.obs)
    # assumes the latlon format from gbif observation building
    def latlon_2_idx(self, latlon):
        y, x =  ~self.affine * (latlon[1], latlon[0])
        return int(round(x)), int(round(y))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get raster data
        lat_lon = self.obs[idx, lat_lon_idx]
        env_rasters = get_raster_point_obs(lat_lon, self.affine, self.rasters, self.nan, self.normalize, self.lat_min, self.lat_max, self.lon_min, self.lon_max)
        # get labels
        specs_label, gens_label, fams_label = get_labels(self.observation, self.obs, idx)
        return (specs_label, gens_label, fams_label, env_rasters)
    def infer_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # obs is of shape [id, species_id, genus, family]    
        # get raster data
        lat_lon = self.obs[idx, lat_lon_idx]
        env_rasters = get_raster_point_obs(lat_lon, self.affine, self.rasters, self.nan, self.normalize, self.lat_min, self.lat_max, self.lon_min, self.lon_max)

        specs_label, gens_label, fams_label, all_spec, all_gen, all_fam = get_inference_labels(self.observation, self.obs, idx)
        return (specs_label, gens_label, fams_label, all_spec, all_gen, all_fam, env_rasters)
    
    # just the environmental rasters as an image
class Bioclim_Rasters_Image(Dataset):
    def __init__(self, base_dir, organism, region, normalize, observation, pix_res=256):
        self.base_dir = base_dir
        self.region = region
        self.organism = organism
        self.normalize = normalize
        self.pix_res = pix_res
        self.observation = observation
        obs = get_gbif_observations(base_dir,organism, region, observation)
        rasterpath = "{}rasters".format(self.base_dir)
        self.rasters, self.affine, obs, self.nan = get_bioclim_rasters(base_dir, region, normalize, obs)
        obs.fillna('nan', inplace=True)               
        if 'species' not in obs.columns:
            obs = utils.add_taxon_metadata(self.base_dir, obs, self.organism)

        obs, inv_spec, spec_dict, gen_dict, fam_dict  = prep_data(obs, observation)
        self.idx_2_id = inv_spec
        # Grab only obs id, species id, genus, family because lat /lon not necessary at the moment
        self.num_specs = len(spec_dict)
        self.num_fams = len(fam_dict)
        self.num_gens = len(gen_dict)
        if observation == 'joint_single'  or observation == 'single_single':
            all_sps = [sp for ob in obs.all_specs for sp in ob]
            all_gen = [sp for ob in obs.all_gens for sp in ob]
            all_fam = [sp for ob in obs.all_fams for sp in ob]
            self.spec_freqs =Counter(all_sps) 
            self.gen_freqs = Counter(all_gen)
            self.fam_freqs = Counter(all_fam)

        else:
            self.spec_freqs = obs.species_id.value_counts().to_dict()
            self.gen_freqs = obs.genus_id.value_counts().to_dict()
            self.fam_freqs = obs.family_id.value_counts().to_dict()
        self.num_rasters = self.rasters.shape[0]
        print("num rasters is ", self.num_rasters)
        self.channels = self.num_rasters
        # convert to numpy
        self.obs = obs[['id', 'species_id', 'genus_id', 'family_id', 'all_specs', 'all_fams', 'all_gens', 'lat_lon']].values

    def __len__(self):
        return len(self.obs)
    # assumes the latlon format from gbif observation building
    def latlon_2_idx(self, latlon):
        y, x =  ~self.affine * (latlon[1], latlon[0])
        return int(round(x)), int(round(y))
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get raster data
        lat_lon = self.obs[idx, lat_lon_idx]
        env_rasters = get_raster_image_obs(lat_lon, self.affine, self.rasters, self.nan, self.normalize, self.pix_res)
        # get labels
        specs_label, gens_label, fams_label = get_labels(self.observation, self.obs, idx)
        return (specs_label, gens_label, fams_label, env_rasters)
    def infer_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        lat_lon = self.obs[idx, lat_lon_idx]
        env_rasters = get_raster_image_obs(lat_lon, self.affine, self.rasters, self.nan, self.normalize, self.pix_res)

        specs_label, gens_label, fams_label, all_spec, all_gen, all_fam = get_inference_labels(self.observation, self.obs, idx)
        return (specs_label, gens_label, fams_label, all_spec, all_gen, all_fam, env_rasters)    
    
class HighRes_Satellite_Rasters_LowRes(Dataset):
    def __init__(self, base_dir, organism, region, normalize, observation, altitude):
        self.base_dir = base_dir
        self.region = region
        self.organism = organism
        self.altitude = altitude
        self.channels = None
        self.normalize = normalize
        self.observation = observation
        obs = get_gbif_observations(base_dir,organism, region, observation)
        rasterpath = "{}rasters".format(self.base_dir)
        self.rasters, self.affine, obs, self.nan = get_bioclim_rasters(base_dir, region, normalize, obs)
        obs.fillna('nan', inplace=True)               
        if 'species' not in obs.columns:
            obs = utils.add_taxon_metadata(self.base_dir, obs, self.organism)

        obs, inv_spec, spec_dict, gen_dict, fam_dict  = prep_data(obs, observation)
        self.idx_2_id = inv_spec
        # Grab only obs id, species id, genus, family because lat /lon not necessary at the moment
        self.num_specs = len(spec_dict)
        self.num_fams = len(fam_dict)
        self.num_gens = len(gen_dict)
        if observation == 'joint_single'  or observation == 'single_single':
            all_sps = [sp for ob in obs.all_specs for sp in ob]
            all_gen = [sp for ob in obs.all_gens for sp in ob]
            all_fam = [sp for ob in obs.all_fams for sp in ob]
            self.spec_freqs =Counter(all_sps) 
            self.gen_freqs = Counter(all_gen)
            self.fam_freqs = Counter(all_fam)

        else:
            self.spec_freqs = obs.species_id.value_counts().to_dict()
            self.gen_freqs = obs.genus_id.value_counts().to_dict()
            self.fam_freqs = obs.family_id.value_counts().to_dict()
            
        self.num_rasters = self.rasters.shape[0] # plus two because including the lat lon
        print("num rasters is ", self.num_rasters)

        # convert to numpy
        self.obs = obs[['id', 'species_id', 'genus_id', 'family_id', 'all_specs', 'all_fams', 'all_gens', 'lat_lon']].values
        channels, width, height = get_shapes(self.obs[0,0], self.base_dir, self.altitude)
        self.pix_res = width
        assert width == height, "the width and height of input images dont match!"
        self.channels = channels + self.num_rasters
        self.width = width
        self.height = height

        
    def __len__(self):
        return len(self.obs)
    # assumes the latlon format from gbif observation building
    def latlon_2_idx(self, latlon):
        y, x =  ~self.affine * (latlon[1], latlon[0])
        return int(round(x)), int(round(y))
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get images  
        id_ = self.obs[idx, id_idx]
        images = image_from_id(id_, self.base_dir, self.altitude)
        # get raster data
        lat_lon = self.obs[idx, lat_lon_idx]
        env_rasters = get_raster_image_obs(lat_lon, self.affine, self.rasters, self.nan, self.normalize, self.width)
        # concatenate together
        all_imgs = np.concatenate([images, env_rasters], axis=0)
        
        # get labels
        specs_label, gens_label, fams_label = get_labels(self.observation, self.obs, idx)
        return (specs_label, gens_label, fams_label, all_imgs)
    def infer_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get images  
        id_ = self.obs[idx, id_idx]
        images = image_from_id(id_, self.base_dir, self.altitude)
        # get raster data
        lat_lon = self.obs[idx, lat_lon_idx]
        env_rasters = get_raster_image_obs(lat_lon, self.affine, self.rasters, self.nan, self.normalize, self.width)
        # concatenate together
        all_imgs = np.concatenate([images, env_rasters], axis=0)
        

        specs_label, gens_label, fams_label, all_spec, all_gen, all_fam = get_inference_labels(self.observation, self.obs, idx)
        return (specs_label, gens_label, fams_label, all_spec, all_gen, all_fam, all_imgs)

class HighRes_Satellite_Rasters_Sheet(Dataset):
    def __init__(self, base_dir, organism, region, normalize, observation, altitude):
        self.base_dir = base_dir
        self.region = region
        self.organism = organism
        self.channels = None
        self.normalize = normalize
        self.altitude = altitude
        self.observation = observation
        obs = get_gbif_observations(base_dir,organism, region, observation)
        rasterpath = "{}rasters".format(self.base_dir)
        self.rasters, self.affine, obs, self.nan = get_bioclim_rasters(base_dir, region, normalize, obs)
        obs.fillna('nan', inplace=True)               
        if 'species' not in obs.columns:
            obs = utils.add_taxon_metadata(self.base_dir, obs, self.organism)

        obs, inv_spec, spec_dict, gen_dict, fam_dict  = prep_data(obs, observation)
        self.idx_2_id = inv_spec
        # Grab only obs id, species id, genus, family because lat /lon not necessary at the moment
        self.num_specs = len(spec_dict)
        self.num_fams = len(fam_dict)
        self.num_gens = len(gen_dict)
        if observation == 'joint_single'  or observation == 'single_single':
            all_sps = [sp for ob in obs.all_specs for sp in ob]
            all_gen = [sp for ob in obs.all_gens for sp in ob]
            all_fam = [sp for ob in obs.all_fams for sp in ob]
            self.spec_freqs =Counter(all_sps) 
            self.gen_freqs = Counter(all_gen)
            self.fam_freqs = Counter(all_fam)

        else:
            self.spec_freqs = obs.species_id.value_counts().to_dict()
            self.gen_freqs = obs.genus_id.value_counts().to_dict()
            self.fam_freqs = obs.family_id.value_counts().to_dict()
            
        self.lat_max = obs.lat.max()
        self.lon_max = obs.lon.max()
        self.lat_min = obs.lat.min()
        self.lon_min = obs.lon.min()        
        self.num_rasters = self.rasters.shape[0]+ 2 # plus two because including the lat lon
        print("num rasters is ", self.num_rasters)

        # convert to numpy
        self.obs = obs[['id', 'species_id', 'genus_id', 'family_id', 'all_specs', 'all_fams', 'all_gens', 'lat_lon']].values
        channels, width, height = get_shapes(self.obs[0,0], self.base_dir, self.altitude)
        self.pix_res = width
        assert width == height, "the width and height of input images dont match!"
        self.channels = channels + self.num_rasters
        self.width = width
        self.height = height

        
    def __len__(self):
        return len(self.obs)
    # assumes the latlon format from gbif observation building
    def latlon_2_idx(self, latlon):
        y, x =  ~self.affine * (latlon[1], latlon[0])
        return int(round(x)), int(round(y))
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get images  
        id_ = self.obs[idx, id_idx]
        images = image_from_id(id_, self.base_dir, self.altitude)
        # get raster data
        lat_lon = self.obs[idx, lat_lon_idx]
        env_rasters = get_raster_sheet_obs(lat_lon, self.affine, self.rasters, self.nan, self.normalize, self.lat_min, self.lat_max, self.lon_min, self.lon_max, self.width, self.height)
        all_imgs = np.concatenate([images, env_rasters], axis=0)
        # get labels
        specs_label, gens_label, fams_label = get_labels(self.observation, self.obs, idx)
        return (specs_label, gens_label, fams_label, all_imgs)
    def infer_item(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # get images  
        id_ = self.obs[idx, id_idx]
        images = image_from_id(id_, self.base_dir, self.altitude)
        # get raster data
        lat_lon = self.obs[idx, lat_lon_idx]
        env_rasters = get_raster_sheet_obs(lat_lon, self.affine, self.rasters, self.nan, self.normalize, self.lat_min, self.lat_max, self.lon_min, self.lon_max, self.width, self.height)
        all_imgs = np.concatenate([images, env_rasters], axis=0)

        specs_label, gens_label, fams_label, all_spec, all_gen, all_fam = get_inference_labels(self.observation, self.obs, idx)
        return (specs_label, gens_label, fams_label, all_spec, all_gen, all_fam, all_imgs)