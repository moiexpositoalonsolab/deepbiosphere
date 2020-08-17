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
    'AZURE_DIR' : '/home/leg/deepbiosphere/GeoCLEF/',
    'MNT_DIR' : '/mnt/GeoCLEF/',
    'MEMEX_LUSTRE' : "/lustre/scratch/lgillespie/",
    'CALC_SCRATCH' : "/NOBACKUP/scratch/lgillespie/"
}
paths = SimpleNamespace(**paths)


choices = {
    'base_dir': ['DBS_DIR', 'MNT_DIR', 'MEMEX_LUSTRE', 'CALC_SCRATCH', 'AZURE_DIR'],
    'region': ['us', 'fr', 'us_fr', 'cali'],
    'organism': ['plant', 'animal', 'plantanimal'],
    'observation': ['single', 'joint_image', 'joint_image_env', 'joint_env_pt', 'joint_env_cnn'],
    'loss' : ['all', 'cumulative', 'sequential', 'just_fam', 'fam_gen', 'none', 'spec_only'],
    'model': ['SkipNet', 'SkipFCNet', 'OGNet', 'OGNoFamNet', 'RandomForest', 'SVM', 'FCNet', 'MixNet', 'SkipFullFamNet', 'MixFullNet']    
}
choices = SimpleNamespace(**choices)

    p.add_argument('--argument', required=False)
    p.add_argument('-a', required='--argument' in sys.argv) #only required if --argument is given
    p.add_argument('-b', required='--argument' in sys.argv) #only required if --argument is given




arguments = {
    # required arguments
    #TODO: convert from_scratch to load_from_config
    'load_from_config' : {'type':str, 'help':"set this option & provide filename to config if you want to run from config", 'required' : False},
    'base_dir': {'type':str, 'help':"what folder to read images from",'choices':choices.base_dir, 'required':True},
    'lr': {'type':float, 'help':"learning rate of model",'required':True},
    'epoch': {'type':int, 'required':True, 'help':"how many epochs to train the model",'required':True},
    'device': {'type':int, 'help':"which gpu to send model to, set -1 for cpu",'required':True},
    'exp_id': {'type':str, 'help':"experiment id of this run", 'required':True},
    # only required if not loading from config
    'region': {'type':str, 'help':"which region to train on", 'choices':choices.region, 'required': ('--load_from_config' in sys.argv) },
    'organism': {'help':"what dataset of what organisms to look at", 'choices':choices.organism,'required': ('--load_from_config' in sys.argv)},
    'batch_size': {'type':int, 'help':"size of batches to use",'required': ('--load_from_config' in sys.argv)}, 
    'observation': {'choices':choices.observation, 'required': ('--load_from_config' in sys.argv)},
    'model':{'choices':choices.model, 'required': ('--load_from_config' in sys.argv)},
    'loss': {'loss':choices.loss, 'required': ('--load_from_config' in sys.argv)}
    # optional arguments
    'processes': {'type':int, 'help':"how many worker processes to use for data loading",'default':1},    
    'seed': {'type':int, 'help':"random seed to use"},
#     'from_scratch':{'dest':"from_scratch", 'help':"start training model from scratch or latest checkpoint", 'action':'store_true'},
    'toy_dataset': {'dest':'toy_dataset', 'help': 'to use a small subset of data, set this option', 'action':'store_true'},
    'dynamic_batch': {'dest':'dynamic_batch', 'help': 'use dynamic sizing of batches', 'action':'store_true'},
    'normalize': {'dest':'normalize', 'help': 'whether to normalize environmental rasters', 'action':'store_true'},
    'weighted': {'dest':'weighted', 'help': 'whether to weight loss by frequency of the observation', 'action':'store_true'},    
    'from_scratch': {'dest':'from_scratch', 'help': 'if you want to restart training from scratch and rebuild everything, set this flag', 'action':'store_true'},    
}

def setup_main_dirs(base_dir):
    '''sets up output, nets, and param directories for saving results to'''
    if not os.path.exists("{}configs/".format(base_dir)):
        os.makedirs("{}configs/".format(base_dir))
    if not os.path.exists("{}nets/".format(base_dir)):
        os.makedirs("{}nets/".format(base_dir))
    if not os.path.exists("{}desiderata/".format(base_dir)):
        os.makedirs("{}desiderata/".format(base_dir))  

def build_params_path(base_dir, observation, organism, region, model, loss, exp_id):
    return "{}configs/{}_{}_{}_{}_{}_{}.json".format(base_dir, observation, organism, region, model, loss, exp_id)

def build_hyperparams_path(base_dir, exp_id):
    if not os.path.exists("{}desiderata/hyperparams/".format(base_dir)):
        os.makedirs("{}desiderata/hyperparams/".format(base_dir))
    return "{}desiderata/hyperparams/{}_{}_{}_{}.json".format(base_dir, datetime.now().day, datetime.now().month, datetime.now().year, exp_id)


class Run_Params():
    def __init__(self, base_dir, ARGS=None, cfg_path=None, device='cpu'):
        
        if abs_path is not None and base_dir is None:
            raise RuntimeError("need both a path and a base directory to open params")
        if ARGS is None and abs_path is None:
            raise RuntimeError("need to specify either command line args or a path to a config file!")
            
        if abs_path is not None or hasattr(ARGS, 'load_from_config'):
            abs_path = "{}configs/{}".format(base_dir, cfg_path)
            assert os.path.exists(cfg_path), "this config doesn't exist on the system! If you would like to rebuild it, please provide CLI to do so"
            print("loading param configs from {}".format(ARGS.load_from_config))
            with open(ARGS.load_from_config) as fp:
                params = json.load(fp)
                params['device'] = ARGS.device                
                self.params = SimpleNamespace(**params)
                self.base_dir = ARGS.base_dir                
        else:
            cfg_path = build_params_path(ARGS.base_dir, ARGS.observation, ARGS.organism, ARGS.region, ARGS.model, ARGS.loss, ARGS.exp_id) 
            params = {
                'lr': ARGS.lr,
                'observation': ARGS.observation,
                'organism' : ARGS.organism,
                'region' : ARGS.region,
                'model' : ARGS.model,
                'exp_id' : ARGS.exp_id,
                'seed' : ARGS.seed,
                'batch_size' : ARGS.batch_size,
                'loss' : ARGS.loss,
            }

            with open(cfg_path, 'w') as fp:
                json.dump(params, fp)
            params['device'] = ARGS.device
            self.params = SimpleNamespace(**params)
            self.base_dir = ARGS.base_dir
#             print(self.params.device, ARGS.device, " hello")
            self.setup_run_dirs(ARGS.base_dir)

    def build_rel_path(self, base_dir, name):
        return "{}{}/{}/{}/{}/{}/{}/".format(base_dir, name, self.params.observation, self.params.organism, self.params.region, self.params.model, self.params.loss)

    def build_abs_nets_path(self, epoch):
        "{}{}_lr{}_e{}.tar".format(self.build_rel_path('nets'), self.params.exp_id, self.params.lr, epoch)

    def build_abs_desider_path(self, epoch):
        return "{}{}_lr{}_e{}.pkl".format(self.build_rel_path('desiderata'), self.params.exp_id, self.params.lr, epoch)
        
    def get_all_models(self):
        paths = "{}{}_lr{}_e{}.tar".format(self.build_rel_path('nets'), self.params.exp_id, self.params.lr, '*')
        epochs = utils.strip_to_epoch(paths)
        return paths, epochs
        
    def get_all_desi(self, base_dir):
        paths =  "{}{}_lr{}_e{}.pkl".format(self.build_rel_path('desiderata'), self.params.exp_id, self.params.lr, "*")
        epochs = utils.strip_to_epoch(paths)
        return paths, epochs

    def get_recent_model(self, epoch=None):
        models, epochs = self.get_all_models('nets')
        all_models = glob.glob(models)
        
        if epoch is not None:
            assert epoch in epochs, "incorrect epoch for this model!"
        if len(all_models) <= 0:
            print("no models for this config on disk")
            return None
        sorted_mods = utils.sort_by_epoch(all_models)
        if epoch is None:
            most_recent = sorted_mods[-1]
            return torch.load(most_recent, map_location=self.device)
        else:
            print("grabbing model at {}".format(epoch))
            for m in sorted_mods:
                if "_e{}".format(epoch) in m:
                    return torch.load(most_recent, map_location=self.device)
        raise RuntimeError("no adequate model found!")


    def get_most_recent_des(self, epoch=None):
        paths, epochs = self.get_all_models('nets')
        desid = glob.glob(paths)
        if epoch is not None:
            assert epoch in epochs, "incorrect epoch for this model!"
        if len(desid) <= 0:
            print("no models for this config on disk")
            return None
        sorted_desi = utils.sort_by_epoch(desid)
        if epoch is None:
            most_recent = sorted_desi[-1]
        else: 
            print("grabbing desiderata at {}".format(epoch))
            for m in sorted_desi:
                if "_e{}".format(epoch) in m:
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
    print("has attr ", hasattr(ARGS, 'seed'))
    if hasattr(ARGS, 'seed') and  ARGS.seed is not None:
        np.random.seed(ARGS.seed)
        torch.manual_seed(ARGS.seed)
    return ARGS