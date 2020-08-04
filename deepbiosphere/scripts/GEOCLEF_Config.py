from datetime import datetime
import numpy as np
import json
import torch
import argparse
import glob
import os
import deepdish as dd
from types import SimpleNamespace 


paths = {
    'DBS_DIR' : "/Carnegie/DPB/Data/Shared/Labs/Moi/Everyone/deepbiosphere/GeoCELF2020/",
    'AZURE_DIR' : '/data/deepbiosphere/deepbiosphere/GeoCLEF/',
    'MNT_DIR' : '/mnt/GeoCLEF/',
    'MEMEX_LUSTRE' : "/lustre/scratch/lgillespie/",
    'CALC_SCRATCH' : "/NOBACKUP/scratch/lgillespie/"
}
paths = SimpleNamespace(**paths)


choices = {
    'base_dir': ['DBS_DIR', 'MNT_DIR', 'MEMEX_LUSTRE', 'CALC_SCRATCH', 'AZURE_DIR'],
    'region': ['us', 'fr', 'us_fr', 'cali'],
    'organism': ['plant', 'animal', 'plantanimal'],
    'observation': ['single', 'joint'],
    'model': ['SkipNet', 'SkipFCNet', 'OGNet', 'OGNoFamNet', 'RandomForest', 'SVM', 'FCNet']    
}
choices = SimpleNamespace(**choices)

arguments = {
    'base_dir': {'type':str, 'help':"what folder to read images from",'choices':choices.base_dir, 'required':True},
    'lr': {'type':float, 'help':"learning rate of model",'required':True},
    'epoch': {'type':int, 'required':True, 'help':"how many epochs to train the model",'required':True},
    'device': {'type':int, 'help':"which gpu to send model to, leave blank for cpu",'required':True},
    'processes': {'type':int, 'help':"how many worker processes to use for data loading",'default':1},
    'exp_id': {'type':str, 'help':"experiment id of this run", 'required':True},
    'region': {'type':str, 'help':"which region to train on", 'required':True, 'choices':choices.region},
    'organism': {'help':"what dataset of what organisms to look at", 'choices':choices.organism,'required':True},
    'seed': {'type':int, 'help':"random seed to use"},
    'GeoCLEF_validate': {'dest':'GeoCLEF_validate', 'help':"whether to validate on GeoClEF validation data or subset of train data", 'action':'store_true'},
    'batch_size': {'type':int, 'help':"size of batches to use",'required':True}, 
    'observation': {'choices':choices.observation, 'required':True},
    'model':{'choices':choices.model, 'required':True},
    'from_scratch':{'dest':"from_scratch", 'help':"start training model from scratch or latest checkpoint", 'action':'store_true'},
    'toy_dataset': {'dest':'toy_dataset', 'help': 'to use a small subset of data, set this option', 'action':'store_true'},
   'dynamic_batch': {'dest':'dynamic_batch', 'help': 'use dynamic sizing of batches'}    
}

def setup_main_dirs(base_dir):
    '''sets up output, nets, and param directories for saving results to'''
    if not os.path.exists("{}configs/".format(base_dir)):
        os.makedirs("{}configs/".format(base_dir))
    if not os.path.exists("{}nets/".format(base_dir)):
        os.makedirs("{}nets/".format(base_dir))
    if not os.path.exists("{}desiderata/".format(base_dir)):
        os.makedirs("{}desiderata/".format(base_dir))  

def build_params_path(base_dir, observation, organism, region, model, exp_id):
    return "{}configs/{}_{}_{}_{}_{}.json".format(base_dir, observation, organism, region, model, exp_id)

def build_hyperparams_path(base_dir, exp_id):
    if not os.path.exists("{}desiderata/hyperparams/".format(base_dir)):
        os.makedirs("{}desiderata/hyperparams/".format(base_dir))
    return "{}desiderata/hyperparams/{}_{}_{}_{}.h5".format(base_dir, datetime.now().day, datetime.now().month, datetime.now().year, exp_id)


class Run_Params():
    def __init__(self, ARGS):
        cfg_path = build_params_path(ARGS.base_dir, ARGS.observation, ARGS.organism, ARGS.region, ARGS.model, ARGS.exp_id)
        if os.path.exists(cfg_path) and not ARGS.from_scratch:
            print("loading param configs from {}".format(cfg_path))
            with open(cfg_path) as fp:
                params = json.load(fp)
                params['device'] = ARGS.device                
                self.params = SimpleNamespace(**params)
        else:
            params = {
                'lr': ARGS.lr,
                'observation': ARGS.observation,
                'organism' : ARGS.organism,
                'region' : ARGS.region,
                'model' : ARGS.model,
                'exp_id' : ARGS.exp_id,
                'seed' : ARGS.seed,
                'batch_size' : ARGS.batch_size
            }

            with open(cfg_path, 'w') as fp:
                json.dump(params, fp)
            params['device'] = ARGS.device
            self.params = SimpleNamespace(**params)
            print(self.params.device, ARGS.device, " hello")
            self.setup_run_dirs(ARGS.base_dir)

    def build_abs_datum_path(self, base_dir, datum, epoch):
        return "{}{}/{}/{}/{}/{}/{}_lr{}_e{}.h5".format(base_dir, datum, self.params.observation, self.params.organism, self.params.region, self.params.model, self.params.exp_id, str(self.params.lr).split(".")[1], epoch)

    def build_datum_path(self, base_dir, datum):
        return "{}{}/{}/{}/{}/{}/".format(base_dir, datum, self.params.observation, self.params.organism, self.params.region, self.params.model)

    def get_recent_model(self, base_dir):
        model_paths = self.build_datum_path(base_dir, 'nets')
        all_models = glob.glob(model_paths + "*")
        if len(all_models) <= 0:
            return None
        else:
            most_recent = sorted(all_models, reverse=True)[0]
            return most_recent
        

    def setup_run_dirs(self, base_dir):
        nets_path = self.build_datum_path(base_dir, 'nets') 
        cfg_path = self.build_datum_path(base_dir, 'desiderata')
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
    if ARGS.seed is not None:
        np.random.seed(ARGS.seed)
        torch.manual_seed(ARGS.seed)
    return ARGS
