import shutil
import json
import glob
import os
import torch
from types import SimpleNamespace
from deepbiosphere.scripts.GEOCLEF_Config import choices, paths, Run_Params
import deepbiosphere.scripts.GEOCLEF_Config as config
import deepbiosphere.scripts.GEOCLEF_Utils as utils
from deepbiosphere.scripts.GEOCLEF_Run import train_model
from itertools import product
import json

''' this script goes in and cleans out all but the last 5 epochs of both nets and desiderata of all models provided'''

def clean_all_models(base_dir, data='nets', num_2_keep=5):
    # https://stackoverflow.com/questions/16953842/using-os-walk-to-recursively-traverse-directories-in-python
    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk("{}{}/".format(base_dir, data)):
        path = root.split(os.sep)
        print((len(path) - 1) * '---', os.path.basename(root))
        print(len(files))
#         print(root, dirs)
        # unique files are based on lr, e
        unq_runs= {file.split("_e")[0] for file in files}
        # so this is one entry per run
        # list of all epochs per run
        for run in unq_runs:
            pths = glob.glob(root+ "/"+ run + "_e*.tar")
            srted = utils.sort_by_epoch(pths)
            to_keep = srted[-num_2_keep:]
            to_toss = srted[:-num_2_keep]
            assert len(to_keep) > 0, "missing models!"
            if len(to_toss) > 0:
                print("removing epochs {} to {} and keeping epochs {} to {} of model {}".format(
                    utils.strip_to_epoch([to_toss[0]])[0], 
                    utils.strip_to_epoch([to_toss[-1]])[0],
                    utils.strip_to_epoch([to_keep[0]])[0], 
                    utils.strip_to_epoch([to_keep[-1]])[0],
                    utils.path_to_cfgname(run)))
                for to_remove in to_toss:
                    os.remove(to_remove)
        print("\n")
# def clean_up_configs(base_dir):
#     # if a config doesn't have any desiderata or nets associated with it, then it's probably junk and can be removed
#     all_cfgs = glob.glob(base_dir + 'configs/*')
#     for cfg_pth in all_cfgs:
#         cfg_name = cfg_pth.split('/')[-1]
#         cfg = Run_Params(base_dir=base_dir, cfg_path=cfg_name)
#         des, _ = cfg.get_all_desi(cleaning=True)
#         mods, _ = cfg.get_all_models(cleaning=True)
#         if len(des) < 1 and len(mods) < 1:
#             print("config {} has no saved files, removing".format(utils.path_to_cfgname(cfg_pth)))
#             abs_pth = cfg.get_cfg_path()
#             os.remove(abs_pth)

def clean_up_configs(base_dir):
    # if a config doesn't have any desiderata or nets associated with it, then it's probably junk and can be removed
    all_cfgs = glob.glob(base_dir + 'configs/*')
    for cfg_pth in all_cfgs:
        cfg_name = cfg_pth.split('/')[-1]
        cfg = Run_Params(base_dir=base_dir, cfg_path=cfg_name)
        try:
            cfg.get_all_desi(cleaning=True)
        except:
            with open(cfg_pth, 'r') as f:
                meme = json.load(f)
                old_nets = "{}{}/{}/{}/{}/{}/{}_lr{}_e*".format(base_dir, 'nets', meme['observation'], meme['organism'], meme['region'], meme['model'], meme['exp_id'], meme['lr'])
                old_desi = "{}{}/{}/{}/{}/{}/{}_lr{}_e*".format(base_dir, 'nets', meme['observation'], meme['organism'], meme['region'], meme['model'], meme['exp_id'], meme['lr'])                
                all_net = glob.glob(old_nets)
                all_des = glob.glob(old_desi)
                print("num nets {} num desiderata {}".format(len(all_net), len(all_des)))
                if len(all_net) < 1 and len(all_des) < 1:
                    print("+++ config {} has no saved files, removing".format(cfg_pth))
                    os.remove(cfg_pth)
            continue
        des, _ = cfg.get_all_desi(cleaning=True)
        mods, _ = cfg.get_all_models(cleaning=True)
        print("num nets {} num desiderata {}".format(len(all_net), len(all_des)))
        if len(des) < 1 and len(mods) < 1:
            print("+++ config {} has no saved files, removing".format(utils.path_to_cfgname(cfg_pth)))
            abs_pth = cfg.get_cfg_path()
            os.remove(abs_pth)               
            
def clean_tensorboard(runs_dir):
    tf_pth = '{}runs/[!r]*/*'.format(runs_dir)
    all_of_em = glob.glob(tf_pth)
    num_bad = 0
    for each in all_of_em:
        epoch = 0   
        try:
            [None for summary in summary_iterator(each)]
        except:
            print("tensorboard record {} corrupted, removing".format(each.split("mod-")[1].split('events')[0]))
            shutil.rmtree(each.split("events")[0])
            num_bad +=1         
            continue
        for summary in summary_iterator(each):        
            if len(summary.summary.value) < 1:
                continue    
            for value in summary.summary.value:
                if 'test' in value.tag:
                    epoch += 1
        if epoch < 1:
            print("removing tensorboard record of {}".format(each.split("mod-")[1].split('/')[0]))
            shutil.rmtree(each.split("events")[0])
            num_bad += 1
    print("number bad is {} out of {}".format(num_bad, len(all_of_em)))

    
def main():
    

    
    #if ARGS.clean_all:
    #    clean_all_models(ARGS.base_dir)
    #else:
    #    params = Run_Params(ARGS.base_dir, ARGS)        
    #    params.clean_old_models()
    clean_tensorboard(ARGS.base_dir.split('GeoCELF2020/')[0])


if __name__ == "__main__":
    # add CLI to nuke a whole model
    # add CLI to clean up all model files

    
    
    args = ['load_from_config', 'loss', 'exp_id', 'base_dir', 'region', 'organism', 'observation','dataset', 'normalize', 'no_alt', 'clean_all']
    ARGS = config.parse_known_args(args)
    main()
