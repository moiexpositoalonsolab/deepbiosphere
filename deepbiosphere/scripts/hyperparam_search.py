from multiprocessing import Process
import torch
from types import SimpleNamespace
from deepbiosphere.scripts.GEOCLEF_Config import choices, paths, Run_Params
import deepbiosphere.scripts.GEOCLEF_Config as config
from deepbiosphere.scripts.GEOCLEF_Run import train_model
from itertools import product
import json

lrs = [
    .00001,
#     .000025,
#     .00005,
#     .000075,
    .0001,
#     .00025,
    .0005,
    .00075,
    .001,
    .0025,
    .005,
#     .0075,    
    .01,
#     .025,
#     .05,
#     .075,        
    .1
]

observation = [
    'joint'
]

organism = [
    'plant'
]
region = [
    'cali'
]

models = ['SkipNet']#, "SkipFCNet"] 

def train_treatment(obs, org, reg, modl, device_num, ARGS):
    
    for lr in lrs:
        # this params isn't the same params as in config
        # this params is just spoofing the normal contents of 
        # ARGS to get around using flags for most of the param args
        params = {
            'lr': lr,
            'observation': obs,
            'organism' : org,
            'region' : reg,
            'model' : modl,
            'exp_id' : ARGS.exp_id,
            'seed' : ARGS.seed,
            'device' : device_num,
            'batch_size' : ARGS.batch_size,
            'base_dir' : ARGS.base_dir,
            'from_scratch' : ARGS.from_scratch
        }
        params = SimpleNamespace(**params)
        params = Run_Params(params)
        print("types in train treat" ,type(ARGS), type(params))
        train_model(ARGS, params)

def main(ARGS):
    # search across models
    # search across lrs
    hp_path = config.build_hyperparams_path(ARGS.base_dir, ARGS.exp_id)
    with open(hp_path, 'w') as fp:
        json.dump({
        'lrs': lrs,
        'observation': observation,
        'organism': organism,
        'region': region,
        'models': models
    }, fp)

    #grab available gpus
    num_gpus = torch.cuda.device_count()
    all_treatments = [observation, organism, region, models]
    num_treatments = len(list(product(*all_treatments)))
    print("num tratments ", num_treatments)
#     num_treatments = len(obs) * len(organism) * len(region) * len(models)
    if num_treatments <= num_gpus:
        processes = []
        for (obs, org, reg, modl), device_num in zip(product(*all_treatments), range(num_treatments)):
            p1 = Process(target = train_treatment(obs, org, reg, modl, device_num, ARGS))
            #TODO: modify train_treatment to loop through lrs and call train_model on each one
            processes.append(p1)
        # TODO: call start on all processes in list
        #TODO: debug this crap
        [p.start() for p in processes]
    else:
        raise NotImplementedError
        # figure out how to evenly break up treatments later

    
if __name__ == "__main__":
    args = ['epoch', 'processes', 'exp_id', 'base_dir', 'region', 'organism', 'seed', 'observation', 'batch_size', 'from_scratch', 'toy_dataset', 'GeoCLEF_validate', 'dynamic_batch']
    ARGS = config.parse_known_args(args)
    config.setup_main_dirs(ARGS.base_dir)
#     params.setup_run_dirs(ARGS.base_dir)

    main(ARGS)    
    
            
    