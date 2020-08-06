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

def get_joint_gbif_data(pth, country, organism):
    ## Grab GBIF observation data
    obs_pth = "{}occurrences/joint_obs_{}_{}.csv".format(pth, country, organism)
    joint_obs = pd.read_csv(obs_pth)  
    joint_obs.all_specs = joint_obs.all_specs.apply(lambda x: parse_string_to_int(x))
    joint_obs.all_gens = joint_obs.all_gens.apply(lambda x: parse_string_to_string(x))
    joint_obs.all_fams = joint_obs.all_fams.apply(lambda x: parse_string_to_string(x))
    return joint_obs

def parse_string_to_string(string):
    string = string.replace("{", '').replace("}", "").replace("'", '')
    split = string.split(", ")
    return split

def parse_string_to_int(string):
    string = string.replace("{", '').replace("}", "").replace("'", '')
    split = string.split(", ")
    return [int(s) for s in split]





def subpath_2_img(pth, subpath, id_):
    alt = "{}{}{}_alti.npy".format(pth, subpath, id_)
    rgbd = "{}{}{}.npy".format(pth, subpath, id_)    
    # Necessary because some data corrupted...
    try:
        np_al = np.load(alt)
        np_img = np.load(rgbd)
    except KeyboardInterrupt:
        print("operation cancelled")
        exit(1)
    except:
        print("trouble loading file {}, faking data :(".format(rgbd))
        # magic numbers 173 and 10000000 are first files in both us and fr datasets
        channels, height, width = get_shapes(173, pth) if id_ < 10000000 else get_shapes(10000000, pth)
        np_al = np.zeros([height, width], dtype='uint8') 
        np_img = np.zeros([channels-1, height, width], dtype='uint8')
        np_img = np.transpose(np_img, (1,2,0))
    np_al = np.expand_dims(np_al, 2)
    np_all = np.concatenate((np_al, np_img), axis=2)
    return np.transpose(np_all,(2, 0, 1))

def image_from_id(id_, pth):
    # make sure image and path are for same region
    cdd, ab, cd = utils.id_2_file(id_)
    subpath = "patches_{}/{}/{}/".format('fr', cd, ab) if id_ >= 10000000 else "patches_{}/patches_{}_{}/{}/{}/".format('us', 'us', cdd, cd, ab)
    return subpath_2_img(pth, subpath, id_)

def freq_from_dict(f_dict):
    list(f_dict.items())
    # sort frequency list by species_id (key of dict)
    return [freq for (sp_id, freq) in sorted(list(f_dict.items()), key=lambda x:x[0])]    

def get_shapes(id_, pth):
    tens = image_from_id(id_, pth)
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

#TODO: normalize dataset range to gaussian distribution
def normalize_dataset():
    pass

class GEOCELF_Dataset(Dataset):
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
        self.obs = obs[['id', 'species_id', 'genus', 'family']].values
        self.transform = transform
        channels, alt_shape, rgbd_shape = get_shapes(self.obs[0,0], self.base_dir)
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
        images = image_from_id(id_, self.base_dir)
        composite_label = self.obs[idx, 1:] # get genus, family as well
        if self.transform:
            images = self.transform(images)
        return (composite_label, images)


class GEOCELF_Dataset_Full(Dataset):
    def __init__(self, base_dir, organism, transform=None):

        self.base_dir = base_dir
        self.split = 'train'
        us_obs = get_gbif_data(self.base_dir, self.split, 'us', organism)
        fr_obs = get_gbif_data(self.base_dir, self.split, 'fr', organism)

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
        channels, alt_shape, rgbd_shape = get_shapes(self.obs[0,0], self.base_dir)
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
        images = image_from_id(id_, self.base_dir)
        composite_label = self.obs[idx, 1:] # get genus, family as well
        if self.transform:
            images = self.transform(images)
        return (composite_label, images)    
    
class GEOCELF_Test_Dataset(Dataset):
    def __init__(self, base_dir, organism, country='us', transform=None):
        
        self.base_dir = base_dir
        self.country = country
        self.split = 'test'
        obs = get_gbif_data(self.base_dir, self.split, country, organism)
        self.obs = obs[['id']].values
        _, alt_shape, rgbd_shape = get_shapes(self.obs[0, 0], self.base_dir)
        self.rgbd_shape = rgbd_shape
        self.transform = transform

    def __len__(self):
        return len(self.obs)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        # obs is of shape [id, species_id, genus, family]    
        id_ = self.obs[idx,0]
        images = image_from_id(id_, self.base_dir)        
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
        
        self.obs = obs[['id']].values
        _, alt_shape, rgbd_shape = get_shapes(self.obs[0, 0], self.base_dir)
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
        images = image_from_id(id_, self.base_dir)               
        if self.transform:
            images = self.transform(images)
        return (images, id_)    
    
    
class GEOCELF_Dataset_Joint(Dataset):
    def __init__(self, base_dir, organism, country='us', transform=None):
        self.base_dir = base_dir
        self.country = country
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
        channels, alt_shape, rgbd_shape = get_shapes(self.obs[0,0], self.base_dir)
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
        images = image_from_id(id_, self.base_dir)                    
        specs_label = self.obs[idx, 1]
        gens_label = self.obs[idx, 3]
        fams_label = self.obs[idx, 2]        
        if self.transform:
            images = self.transform(images)
        return (specs_label, gens_label, fams_label, images)    
    

class GEOCELF_Dataset_Joint_Full(Dataset):
    def __init__(self, base_dir, organism, transform=None):
        
        self.base_dir = base_dir
        self.split = 'train'
        self.organism = organism
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
        channels, alt_shape, rgbd_shape = get_shapes(self.obs[0,0], self.base_dir)
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
        images = image_from_id(id_, self.base_dir)            
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
        obs = obs[:500]
        self.obs = obs[['id', 'all_specs', 'all_fams', 'all_gens']].values
        self.transform = transform
        channels, alt_shape, rgbd_shape = get_shapes(self.obs[0,0], self.base_dir)
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
        images = image_from_id(id_, self.base_dir)                    
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
        channels, alt_shape, rgbd_shape = get_shapes(self.obs[0,0], self.base_dir)
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
        images = image_from_id(id_, self.base_dir)
        composite_label = self.obs[idx, 1:] # get genus, family as well
        if self.transform:
            images = self.transform(images)
        return (composite_label, images)
    
    
class GEOCELF_Dataset_Joint_Scalar_Raster(Dataset):
    def __init__(self, base_dir, organism, country='us', transform=None, normalize=True):
        self.base_dir = base_dir
        self.country = country
        self.organism = organism
        obs = get_joint_gbif_data(self.base_dir, country, organism)
        rasterpath = f"{self.base_dir}rasters"
        rasters  = PatchExtractor(rasterpath, size = 1)
        self.rasters = rasters.add_all(normalized=normalize)
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
        channels, alt_shape, rgbd_shape = get_shapes(self.obs[0,0], self.base_dir)
        self.channels = channels
        self.num_rasters = len(self.rasters)
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
        images = image_from_id(id_, self.base_dir)     
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