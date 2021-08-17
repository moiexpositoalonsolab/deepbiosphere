import sys
import pickle
from datetime import datetime
import numpy as np
import json
import torch
import argparse
import glob
import os
import deepbiosphere.scripts.GEOCLEF_Utils as utils
from types import SimpleNamespace


paths = {
    'DBS_DIR' : "/Carnegie/DPB/Data/Shared/Labs/Moi/Everyone/deepbiosphere/GeoCELF2020/",
    'AZURE_DIR' : '/home/leg/deepbiosphere/GeoCELF2020/',
    'NAIP_BASE' : '/home/leg/NAIP/v002/',
    'VIRT_BASE' : '/mnt/vrt/',
    'MNT_DIR' : '/mnt/GeoCLEF/',
    'MEMEX_LUSTRE' : "/lustre/scratch/lgillespie/",
    'CALC_SCRATCH' : "/NOBACKUP/scratch/lgillespie/"
}
paths = SimpleNamespace(**paths)


choices = {
    'base_dir': ['DBS_DIR', 'MNT_DIR', 'MEMEX_LUSTRE', 'CALC_SCRATCH', 'AZURE_DIR'],
    'region': ['us', 'fr', 'us_fr', 'cali'],
    'observation': ['single', 'joint_multiple', 'joint_single', 'single_single'],
    'organism': ['plant', 'animal', 'plantanimal'],
    # single: single obs, joint_image is just trained on the rgbd image, joint_image_env is trained on rgbd image + env cnn rasters,
    # joint_image_pt is trained on rgbd image + env rasters pointwise, joint_env_cnn is trained on just environmental rasters as a cnn
    # joint_pt is trained as just the pointwise env rasters for a given observation
    # old: joint_image', 'joint_image_env', 'joint_image_pt', 'joint_env_cnn', 'joint_pt', 'joint_image_cnn', 'single_image_cnn', 'single_env_cnn'],
    # new: satellite_only, satellite_rasters_lowres, satellite_rasters_point, rasters_lowres, rasters_point, satellite_rasters_sheet,
    'dataset': [ 'satellite_only', 'satellite_rasters_image', 'satellite_rasters_point', 'rasters_image', 'rasters_point', 'satellite_rasters_sheet'],
    # loss is overcoded to both set the actual loss function and to determine when what loss is calculated and for what taxonomy
    'loss' : ['all', 'cumulative', 'sequential', 'just_fam', 'fam_gen', 'none', 'just_spec',
             'MultiLabelMarginLoss', 'BCEWithLogits','BrierPresenceOnly','BrierAll','CrossEntropyPresenceOnly','AsymmetricLoss','AsymmetricLossOptimized'
             ],
    'model': ['SkipNet', 'SkipFCNet', 'OGNet', 'OGNoFamNet', 'SVM', 'FCNet', 'MixNet', 'SkipFullFamNet', 'MixFullNet','SpecOnly', 'MLP_Family', 'MLP_Family_Genus', 'MLP_Family_Genus_Species', 'FlatNet', 'ResNet_18', 'ResNet_34', 'VGG_11',  'VGG_16', 'Joint_VGG11_MLP', 'Joint_VGG16_MLP', 'TResNet_M', 'TResNet_L', 'Joint_TResNet_M','Joint_TResNet_L', 'Joint_ResNet_18', 'New_MLP_Family_Genus_Species', 'RandomForestClassifier', 'MaxEnt'],
    'normalize' : ['normalize', 'min_max', 'none'],
    'loss_type' : ['none', 'mean', 'sum'],
    'arch_type' : ['plain', 'remove_fc', 'scale_fc'],
    'pretrained_dset' : ['imagenet', 'mscoco','joint_multiple_obs_cali_plant_train_4', 'none'],
    'pretrained' : ['none', 'feat_ext', 'finetune'],
    'test_or_train' : ['test_only', 'train_only', 'test_and_train'],
    'which_taxa' : ['spec_only', 'spec_gen_fam', 'gen_fam', 'spec_gen'],
    'ecoregion' : ['NA_L1NAME','NA_L2NAME','NA_L3NAME','US_L3NAME']

}
choices = SimpleNamespace(**choices)


arguments = {
    # required arguments
    #TODO: convert from_scratch to load_from_config
    'load_from_config' : {'type':str, 'help':"set this option & provide filename to config if you want to run from config", 'required' : False},
    'base_dir': {'type':str, 'help':"what folder to read images from",'choices':choices.base_dir, 'required':True},
    'lr': {'type':float, 'help':"learning rate of model",'required':True},
    'epoch': {'type':int,'help':"how many epochs to train the model if training and what epoch to run inference on if inferring. To run most recent model for inference, use value of -1.",'default':-1},
    'device': {'type':int, 'help':"which gpu to send model to, set -1 for cpu",'required':True},
    # only required if not loading from config
    'region': {'type':str, 'help':"which region to train on", 'choices':choices.region, 'required': ('--load_from_config' not in sys.argv) },
    'organism': {'help':"what dataset of what organisms to look at", 'choices':choices.organism,'required': ('--load_from_config' not in sys.argv)},
    'exp_id': {'type':str, 'help':"experiment id of this run", 'required':('--load_from_config' not in sys.argv)},
    'batch_size': {'type':int, 'help':"size of batches to use",'required': ('--load_from_config' not in sys.argv)},
    'observation': {'choices':choices.observation, 'required': ('--load_from_config' not in sys.argv)},
    'dataset': {'choices':choices.dataset, 'required': ('--load_from_config' not in sys.argv)},
    'model':{'choices':choices.model, 'required': ('--load_from_config' not in sys.argv)},
    'loss': {'choices':choices.loss, 'required': ('--load_from_config' not in sys.argv)},
    'loss_type': {'choices':choices.loss_type, 'default' : 'mean'},
    'threshold' : {'dest':'threshold', 'type':int,'help' : "how many observations must a species at least have to be added to the dataset", 'default':4},
    'arch_type' : {'choices':choices.arch_type, 'help' : 'which CNN architecture to use for ResNet and VGGNet', 'default' : 'plain'},
    'pretrained' : {'choices' : choices.pretrained, 'help' : 'what kind of pretrained neural network to use (if pretrained at all)', 'default' : 'none'},
    'pretrained_dset' : {'choices' : choices.pretrained_dset, 'help' : 'which dataset you prefer the neural network be pretrained on if pretrained at all and if pretrain dataset is available for the model)', 'default' : 'none'},
    # optional arguments
    'processes': {'type':int, 'help':"how many worker processes to use for data loading",'default':1},
    'seed': {'type':int, 'help':"random seed to use"},
    'toy_dataset': {'dest':'toy_dataset', 'help': 'to use a small subset of data, set this option', 'action':'store_true'},
    #'dynamic_batch': {'dest':'dynamic_batch', 'help': 'use dynamic sizing of batches', 'action':'store_true'},
    'clean_all': {'dest':'clean_all', 'help': 'whether to clean out all old configs and files or not', 'action':'store_true'},
    'normalize': {'choices': choices.normalize, 'help': 'whether to normalize environmental rasters'},
    'unweighted': {'dest':'unweighted', 'help': 'whether to weight loss by frequency of the observation', 'action':'store_true'},
    'no_alt': {'dest':'no_alt', 'help': 'set to not include altitude layer, dont set to include', 'action':'store_false'},
    'from_scratch': {'dest':'from_scratch', 'help': 'if you want to restart training from scratch and rebuild everything, set this flag', 'action':'store_true'},
    'census' : {'dest':'census', 'help' : "use if filtering to the us census raster area", 'action' : 'store_true'},
    'ecoregions_only' : {'dest':'ecoregions_only', 'help' : "use if filtering to the us census raster area", 'action' : 'store_true'},
    'num_species' : {'type' : int, 'help' : 'for building dataset, if want to cut to top K species, set this option', 'default': -1},
    'batch_norm' : {'dest' : 'batch_norm', 'help' : 'whether or not batch norm was used for training the network', 'action' : 'store_true'},
        'test_or_train' : {'choices' : choices.test_or_train, 'help' : 'Run inference on just train, just test, or both', 'default' : 'test_only'},
        'which_taxa' : {'choices' : choices.which_taxa, 'help' : 'Which taxonomic levels to use', 'default' : 'spec_only'},
    'n_trees': {'type':int,'help':"How many trees to build for random forest model",'default':20},
    'config_path': {'type':str,'help':"relative (within base_dir) path to json of models to load",'required' : True},
    'ecoregion': {'choices' : choices.ecoregion,'help':"which ecoregion to split the data into",'default' : 'NA_L3NAME'},
    'pres_threshold': {'type' : float,'help':"what value to threshold the presence-absence of the model",'default' : 0.5},
    'inc_latlon': {'dest' : 'inc_latlon', 'help' : 'whether to exclude the latitude and longitude of an observation for the rasters', 'action' : 'store_true'},
}

def get_res_dir(base_dir):
    dir = "{}results/{}_{}_{}/".format(base_dir, datetime.now().day, datetime.now().month, datetime.now().year)
    if not os.path.exists(dir):
        os.makedirs(dir)
    return dir

def setup_pretrained_dirs(base_dir):
    if not os.path.exists("{}nets/pretrained/".format(base_dir)):
        os.makedirs("{}nets/pretrained/".format(base_dir))
    return "{}nets/pretrained/".format(base_dir)


def setup_main_dirs(base_dir):
    '''sets up output, nets, and param directories for saving results to'''
    if not os.path.exists("{}configs/".format(base_dir)):
        os.makedirs("{}configs/".format(base_dir))
    if not os.path.exists("{}nets/".format(base_dir)):
        os.makedirs("{}nets/".format(base_dir))
    if not os.path.exists("{}desiderata/".format(base_dir)):
        os.makedirs("{}desiderata/".format(base_dir))
    if not os.path.exists("{}inference/".format(base_dir)):
        os.makedirs("{}inference/".format(base_dir))
    if not os.path.exists("{}occurrences/".format(base_dir)):
            os.makedirs("{}occurrences/".format(base_dir))

def build_config_name(observation, organism, region, model, loss, dataset, exp_id):
    return "{}_{}_{}_{}_{}_{}_{}".format(observation, organism, region, model, loss, dataset, exp_id)

def build_gbif_file(taxon, start_date, end_date, area, ext='json'):
    return  "{}_{}_{}_{}.{}".format(taxon, start_date, end_date, area[0].replace('.', '_'), ext)

def build_inference_path(base_dir, model, loss, exp_id, taxa, num_specs, dir=False, across_time=False):

    if dir:
        return "{}inference/".format(base_dir)
    else:
        if num_specs < 0:
            nsp = 'all_spec'
        else:
            nsp = "top_{}_spec".format(num_specs)
        return "{}inference/{}".format(base_dir, build_inference_name(model, loss, exp_id, taxa, nsp, across_time=across_time))
def extract_numspecs(infer_pth):
    name = infer_pth.split('/')[-1]
    import pdb; pdb.set_trace()
    return name.split('_')[4] #TODO: if build_inference_name changes, this must change too
#     "{}_{}_{}_{}_{}_{}_{}_{}.csv".format(model, loss, exp_id, taxa, num_species, datetime.now().day, datetime.now().month, datetime.now().year)

def build_inference_name(model, loss, exp_id, taxa, num_species, across_time=False):
    if across_time:
        return "{}_{}_{}_{}_{}_{}.csv".format(model, loss, exp_id, taxa, num_species, '*')
    else:
        return "{}_{}_{}_{}_{}_{}_{}_{}.csv".format(model, loss, exp_id, taxa, num_species, datetime.now().day, datetime.now().month, datetime.now().year)

def build_params_path(base_dir, observation, organism, region, model, loss, dataset, exp_id, dir=False, across_time=False):
    if dir:
        return "{}configs/".format(base_dir)
    else:
        return "{}configs/{}.json".format(base_dir, build_config_name(observation, organism, region, model, loss, dataset, exp_id))

def build_hyperparams_path(base_dir, exp_id):
    if not os.path.exists("{}desiderata/hyperparams/".format(base_dir)):
        os.makedirs("{}desiderata/hyperparams/".format(base_dir))
    return "{}desiderata/hyperparams/{}_{}_{}_{}.json".format(base_dir, datetime.now().day, datetime.now().month, datetime.now().year, exp_id)

def load_parameters(abs_path):
#     print(abs_path)
    if not os.path.exists(abs_path):
        raise FileNotFoundError("this config {} doesn't exist on the system! If you would like to rebuild it, please provide CLI to do so".format(abs_path))
    print("loading param configs from {}".format(utils.path_to_cfgname(abs_path)))
    with open(abs_path, 'r') as fp:
        params = json.load(fp)
    params = SimpleNamespace(**params)
    return params
def get_ground_truth(num_species, base_dir):
    spec_pth = build_inference_path(base_dir, 'ground_truth', "", 'species', num_species)
    gen_pth = build_inference_path(base_dir, 'ground_truth', '', 'genus', num_species)
    fam_pth = build_inference_path(base_dir, 'ground_truth', '', 'family', num_species)
    return spec_pth, gen_pth, fam_pth

class Run_Params():
    def __init__(self, base_dir, ARGS=None, cfg_path=None):

        if cfg_path is not None and base_dir is None:
            raise RuntimeError("need both a path and a base directory to open params")
        elif ARGS is None and cfg_path is None:
            raise RuntimeError("need to specify either command line args or a path to a config file!")
        if cfg_path is not None:
            abs_path = "{}configs/{}".format(base_dir, cfg_path)
            self.params = load_parameters(abs_path)
            if self.params.model == "RandomForestClassifier":
                print("hello world")
                self.params.loss = self.params.n_trees
            self.base_dir = base_dir
        # will hopefully short circuit out of check if args is None but cfg_path isn't
        elif ARGS.load_from_config is not None :
            abs_path = "{}configs/{}".format(base_dir, ARGS.load_from_config)
            self.params = load_parameters(abs_path)
            if self.params.model == "RandomForestClassifier":
                self.params.loss = self.params.n_trees
            if ARGS.batch_size is not None:
                self.params.batch_size = ARGS.batch_size
            self.base_dir = ARGS.base_dir
        else:
            if ARGS.model == 'RandomForestClassifier':
                loss = ARGS.n_trees
            else:
                loss = ARGS.loss
            cfg_path = build_params_path(ARGS.base_dir, ARGS.observation, ARGS.organism, ARGS.region, ARGS.model, loss, ARGS.dataset, ARGS.exp_id)

            if ARGS.model == 'RandomForestClassifier':
                params = {
                    'observation': ARGS.observation,
                    'organism' : ARGS.organism,
                    'region' : ARGS.region,
                    'model' : ARGS.model,
                    'exp_id' : ARGS.exp_id,
                    'seed' : ARGS.seed,
                    'normalize' : ARGS.normalize,
                    'inc_latlon' : ARGS.inc_latlon,
                    'unweighted' : ARGS.unweighted,
                    'dataset' : ARGS.dataset,
                    'threshold' : ARGS.threshold,
                    'no_altitude' : ARGS.no_alt,
                    'n_trees' : ARGS.n_trees,
                    'loss' : ARGS.n_trees,  # sketchy I know but gets files to work
                }
            elif ARGS.model == 'MaxEnt':
                params = {
                    'observation': ARGS.observation,
                    'organism' : ARGS.organism,
                    'region' : ARGS.region,
                    'model' : ARGS.model,
                    'exp_id' : ARGS.exp_id,
                    'dataset' : ARGS.dataset,
                    'inc_latlon' : ARGS.inc_latlon,
                    'threshold' : ARGS.threshold,
                    'loss' : ARGS.loss,
                    'no_altitude' : ARGS.no_alt,
                    'normalize' : ARGS.normalize,
                }

            else:
                params = {
                    'lr': ARGS.lr,
                    'observation': ARGS.observation,
                    'organism' : ARGS.organism,
                    'region' : ARGS.region,
                    'model' : ARGS.model,
                    'exp_id' : ARGS.exp_id,
                    'seed' : ARGS.seed,
                    'batch_size' : ARGS.batch_size,
                    'inc_latlon' : ARGS.inc_latlon,
                    'loss' : ARGS.loss,
                    'normalize' : ARGS.normalize,
                    'unweighted' : ARGS.unweighted,
                    'no_altitude' : ARGS.no_alt,
                    'dataset' : ARGS.dataset,
                    'threshold' : ARGS.threshold,
                    'loss_type' : ARGS.loss_type,
                    'pretrained' : ARGS.pretrained,
                    'pretrained_dset' : ARGS.pretrained_dset,
                    'batch_norm' : ARGS.batch_norm,
                    'arch_type' : ARGS.arch_type,

                }
            with open(cfg_path, 'w') as fp:
                json.dump(params, fp)
            self.params = SimpleNamespace(**params)
            self.base_dir = ARGS.base_dir
            if ARGS.model != 'RandomForestClassifier':
                self.setup_run_dirs(ARGS.base_dir)

    def build_rel_path(self, base_dir, name):
        return "{}{}/{}/{}/{}/{}/{}/{}/".format(base_dir, name, self.params.observation, self.params.organism, self.params.region, self.params.model, self.params.loss, self.params.dataset)

    def build_abs_nets_path(self, epoch):
        return "{}{}_lr{}_e{}.tar".format(self.build_rel_path(self.base_dir, 'nets'), self.params.exp_id, self.params.lr, epoch)

    def build_abs_desider_path(self, epoch):
        return "{}{}_lr{}_e{}.pkl".format(self.build_rel_path(self.base_dir, 'desiderata'), self.params.exp_id, self.params.lr, epoch)

    def get_all_models(self, cleaning=False):
        paths = "{}{}_lr{}_e{}.tar".format(self.build_rel_path(self.base_dir, 'nets'), self.params.exp_id, self.params.lr, '*')
#         print('get_all_models', paths)
        paths = glob.glob(paths)
        if not cleaning:
            assert len(paths) > 0, 'no models to load in!'
        epochs = utils.strip_to_epoch(paths)
        return paths, epochs

    def get_all_desi(self, cleaning=False):
        paths =  "{}{}_lr{}_e{}.pkl".format(self.build_rel_path(self.base_dir, 'desiderata'), self.params.exp_id, self.params.lr, "*")
        paths = glob.glob(paths)
        if not cleaning:
            assert len(paths) > 0, 'no desiderata to load in!'
        epochs = utils.strip_to_epoch(paths)
        return paths, epochs

    def get_recent_model(self, epoch=None, device='cpu'):
        all_models, epochs = self.get_all_models()
#         print('models ', all_models, epochs)
#         all_models = glob.glob(models)

        if epoch is not None:
            assert epoch in epochs, "incorrect epoch for this model!"
        if len(all_models) <= 0:
            print("no models for this config on disk")
            return None
        sorted_mods = utils.sort_by_epoch(all_models)
        if epoch is None:
            most_recent = sorted_mods[-1]
            print("loading ", most_recent)
            return torch.load(most_recent, map_location=device)
        else:
            print("grabbing model at {}".format(epoch))
            for m in sorted_mods:
                if "_e{}".format(epoch) in m:
                    return torch.load(m, map_location=device)
        print("no adequate model found!")
        return None

    def get_all_inference_specgenfam(self, num_specs):

        pth_spec = build_inference_path(self.base_dir, self.params.model, self.params.loss, self.params.exp_id, 'species', num_specs, across_time=True)
        pth_gen = build_inference_path(self.base_dir, self.params.model, self.params.loss, self.params.exp_id, 'genus', num_specs, across_time=True)
        pth_fam = build_inference_path(self.base_dir, self.params.model, self.params.loss, self.params.exp_id, 'family', num_specs, across_time=True)
        pths_s = glob.glob(pth_spec)
        pths_g = glob.glob(pth_gen)
        pths_f = glob.glob(pth_fam)
        return pths_s, pths_g, pths_f

    def get_all_inference_speconly(self, num_specs):
        pth_spec = build_inference_path(self.base_dir, self.params.model, self.params.loss, self.params.exp_id, 'species', num_specs, across_time=True)
        pths_s = glob.glob(pth_spec)
        return pths_s

        #TODO: doesn't handle bonus stuff properly
    def get_most_recent_inference(self, num_species=-1, which_taxa='spec_gen_fam'):
        if which_taxa == 'spec_gen_fam':
            sp, gen,fam = self.get_all_inference_specgenfam(num_species)
            assert len(sp) > 0 and len(gen)  > 0 and len(fam) > 0, "inference files missing for a taxa category!"
            print("sorting")
            sp.sort(key=os.path.getmtime, reverse=True)
            print("species")
            gen.sort(key=os.path.getmtime, reverse=True)
            print("genus")
            fam.sort(key=os.path.getmtime, reverse=True)
            print("family")
            return sp[0], gen[0], fam[0]
        elif which_taxa == 'spec_only':
            sp = self.get_all_inference_speconly(num_species)
            assert len(sp) > 0 , "inference files missing for a taxa category!"
            print("sorting")
            sp.sort(key=os.path.getmtime, reverse=True)
            print("species")
            return sp[0]
        else:
            raise NotImplementedError("not yet implementedf or ", which_taxa)

    def get_most_recent_des(self, epoch=None):
        paths, epochs = self.get_all_desi()
        if epoch is not None:
            assert epoch in epochs, "incorrect epoch for this model!"
        if len(paths) <= 0:
            print("no models for this config on disk")
            return None
        sorted_desi = utils.sort_by_epoch(paths)
#         print(epoch)
        if epoch is None:
#             print('in none')
            most_recent = sorted_desi[-1]
            with open(most_recent, 'rb') as f:
                des = pickle.load(f)
            return des

        else:
            print("grabbing desiderata at {}".format(epoch))
            for m in sorted_desi:
                if "_e{}".format(epoch) in m:
                    most_recent = m
                    with open(most_recent, 'rb') as f:
                        des = pickle.load(f)
                    return des
        raise RuntimeError("no adequate desiderata found!")

    def get_split(self, epoch=None):
        des = self.get_most_recent_des(epoch)
        return des['splits']['train'], des['splits']['test']

    def setup_run_dirs(self, base_dir):
        nets_path = self.build_rel_path(base_dir, 'nets')
        cfg_path = self.build_rel_path(base_dir, 'desiderata')
        if not os.path.exists(nets_path):
            os.makedirs(nets_path)
        if not os.path.exists(cfg_path):
            os.makedirs(cfg_path)
    def clean_old_models(self, num_2_keep=5):
        paths, e = self.get_all_models()
        srt_pths = utils.sort_by_epoch(paths)
        to_keep = srted[-num_2_keep:]
        to_toss = srted[:-num_2_keep]
        print("removing epochs {} to {}".format(to_toss[0], to_toss[-1]))
        for removed in to_toss:
            os.remove(removed)
    def get_cfg_path(self):
        return build_params_path(self.base_dir, self.params.observation, self.params.organism, self.params.region, self.params.model, self.params.loss, self.params.dataset, self.params.exp_id)

    def get_cfg_name(self):
        return build_config_name(self.params.observation, self.params.organism, self.params.region, self.params.model, self.params.loss, self.params.dataset, self.params.exp_id)

    def remove_config(self):
        print("removing {}".format(self.get_cfg_name()))
        m_pths, _ = self.get_all_models()
        d_pths, _ = self.get_all_desi()
        c_pth = self.get_cfg_path()
        to_remove = m_pths + d_pths + [c_pth]
        for removal in to_remove:
            os.remove(removal)

def parse_known_args(args):
    if args is None:
        exit(1), "no arguments were specificed!"
    parser = argparse.ArgumentParser()
    for arg in args:
        parser.add_argument("--{}".format(arg), **arguments[arg])
    ARGS, _ = parser.parse_known_args()
    # parsing which path to use
    ARGS.base_dir = getattr(paths, ARGS.base_dir)
    print("using base directory {}".format(ARGS.base_dir))
    if hasattr(ARGS, 'seed') and  ARGS.seed is not None:
        np.random.seed(ARGS.seed)
        torch.manual_seed(ARGS.seed)
    return ARGS
