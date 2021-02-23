from datetime import datetime
from tqdm import tqdm
import deepbiosphere.scripts.GEOCLEF_Utils as utils
from deepbiosphere.scripts.GEOCLEF_Utils import torch_intersection, recall_per_example, accuracy_per_example, precision_per_example
from deepbiosphere.scripts import GEOCLEF_CNN as nets
import glob
import torch
import torch.functional as F
from deepbiosphere.scripts.GEOCLEF_Config import paths, Run_Params
from deepbiosphere.scripts import GEOCLEF_Config as config
from deepbiosphere.scripts.GEOCLEF_Run import  setup_dataset, setup_model, setup_loss
import deepbiosphere.scripts.GEOCLEF_Run as run
import deepbiosphere.scripts.GEOCLEF_Dataset as Dataset
#import deepbiosphere.scripts.as inference
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle


#def eval_model_old(params, base_dir, device, test, outputs='all'):
# this inference spits out a csv of the form:
# with one file with raw logits and one file with sigmoid threshold?
# makes one of these for genus and family too?
#     obs    spec1    spec2    ...    specN
#     ob1    .01     .351              .3510
#     ob2    .01     .351              .3510
#     .
#     .
#     .
#     obM    .01     .351              .3510
def eval_model_raw(params, base_dir, device, test, outputs, num_species, processes, epoch):

    
    # always runs inference on the model at oldest epoch, TODO: add functionality to select the epoch to run inference on   
    if epoch == -1:
        state = params.get_recent_model(epoch=None)
    else:
        state = params.get_recent_model(epoch=epoch)
    # get topk and pass in as num_species
    dset= run.setup_dataset(params.params.observation, params.base_dir, params.params.organism, params.params.region, params.params.normalize, params.params.no_altitude, params.params.dataset, params.params.threshold, num_species=num_species)
    train_samp, test_samp, idxs = run.better_split_train_test(dset)

    global num_specs 
    num_specs = dset.num_specs
    global num_fams
    num_fams = dset.num_fams
    global num_gens
    num_gens = dset.num_gens   
    if params.params.dataset == 'satellite_rasters_point':
        collate_fn = joint_raster_collate_fn
    else:
        collate_fn = joint_collate_fn
    train_loader = run.setup_dataloader(dset, params.params.dataset, params.params.batch_size, processes, train_samp, params.params.model, joint_collate_fn=collate_fn)
    test_loader = run.setup_dataloader(dset, params.params.dataset, params.params.batch_size, processes, test_samp, params.params.model, joint_collate_fn=collate_fn)    

    model = run.setup_model(params.params.model, dset, params.params.pretrained, params.params.batch_norm, params.params.arch_type)
#     import pdb;
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device)    
    # again TODO: this is messy and needs to be fixed
    if params.params.pretrained != 'none' and 'TResNet' in params.params.model:
        # change mean subtraction to match what pretrained tresnet expects
        print("WE ARE CHANGING SOME PARAMETERS")
        dset.dataset_means = Dataset.dataset_means['none']

    # TODO: make sure dataframe has all the info it needs for plotting
    obs = Dataset.get_gbif_observations(base_dir, params.params.organism, params.params.region, params.params.observation, params.params.threshold, num_species)
    obs.fillna('nan', inplace=True)
    if 'species' not in obs.columns:
        obs = utils.add_taxon_metadata(self.base_dir, obs, self.organism)

    output_spec = np.full([len(dset), dset.num_specs], np.nan)
    output_gen = np.full([len(dset), dset.num_gens ], np.nan)
    output_fam = np.full([len(dset), dset.num_fams ], np.nan)    
    if test == 'test_only':
        print('inferring on test set')
        output_spec, output_gen, output_fam = run_inference(test_loader, params, model, device, output_spec, output_gen, output_fam)
    elif test == 'train_only':
        print('inferring on train set')        
        output_spec, output_gen, output_fam = run_inference(train_loader, params, model, device, output_spec, output_gen, output_fam)
    elif test == 'test_and_train':
        output_spec, output_gen, output_fam = run_inference(test_loader, params, model, device, output_spec, output_gen, output_fam)
        output_spec, output_gen, output_fam = run_inference(train_loader, params, model, device, output_spec, output_gen, output_fam)
        
        
    print("saving data")
    tick = time.time()

    to_transfer = ['lat', 'lon', 'region', 'city', 'NA_L3NAME', 'US_L3NAME', 'NA_L2NAME', 'NA_L1NAME', 'test']
    inv_gen = {v: k for k, v in dset.gen_dict.items()}
    inv_fam = {v: k for k, v in dset.fam_dict.items()}
    
    df_spec_cols = [dset.inv_spec[i] for i in range(dset.num_specs)]
    df_gen_cols = [inv_gen[i] for i in range(dset.num_gens)]
    df_fam_cols = [inv_fam[i] for i in range(dset.num_fams)]    
    df_spec = utils.numpy_2_df(output_spec, df_spec_cols, obs, to_transfer)
    df_gen = utils.numpy_2_df(output_gen, df_gen_cols, obs, to_transfer)
    df_fam = utils.numpy_2_df(output_fam, df_fam_cols, obs, to_transfer)
    extra_dat = "{}_{}".format(params.params.loss, params.params.exp_id)
    spec_pth = config.build_inference_path(base_dir, params.params.model, extra_dat, 'species', num_species)
    gen_pth = config.build_inference_path(base_dir, params.params.model, extra_dat, 'genus', num_species)
    fam_pth = config.build_inference_path(base_dir, params.params.model, extra_dat, 'family', num_species)  
    df_spec.to_csv(spec_pth)
    df_gen.to_csv(gen_pth)
    df_fam.to_csv(fam_pth)
    tock = time.time()
    print("took {} minutes to save data".format((tock-tick)/60))

def joint_collate_fn(batch):
    # batch is a list of tuples of (specs_label, gens_label, fams_label, images, idx)  
    all_specs = []
    all_gens = []
    all_fams = []
    imgs = []
    idxs = []
    #(specs_label, gens_label, fams_label, images)  
    for (spec, gen, fam, img, idx) in batch:
        specs_tens = torch.zeros(num_specs)
        specs_tens[spec] += 1
        all_specs.append(specs_tens)

        gens_tens = torch.zeros(num_gens)
        gens_tens[gen] += 1
        all_gens.append(gens_tens)

        fams_tens = torch.zeros(num_fams)
        fams_tens[fam] += 1
        all_fams.append(fams_tens)
        imgs.append(img)
        idxs.append(idx)
    return torch.stack(all_specs), torch.stack(all_gens), torch.stack(all_fams), torch.from_numpy(np.stack(imgs)), idxs

def joint_raster_collate_fn(batch):
    # batch is a list of tuples of (specs_label, gens_label, fams_label, images, env_rasters, idx)  
    all_specs = []
    all_gens = []
    all_fams = []
    imgs = []
    rasters = []
    idxs = []
    #(specs_label, gens_label, fams_label, images, env_rasters)  
    for (spec, gen, fam, img, raster, idx) in batch:
        specs_tens = torch.zeros(num_specs)
        specs_tens[spec] += 1
        all_specs.append(specs_tens)

        gens_tens = torch.zeros(num_gens)
        gens_tens[gen] += 1
        all_gens.append(gens_tens)

        fams_tens = torch.zeros(num_fams)
        fams_tens[fam] += 1
        all_fams.append(fams_tens)
        imgs.append(img)
        rasters.append(raster)
        idxs.append(idx)
    return torch.stack(all_specs), torch.stack(all_gens), torch.stack(all_fams), torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(rasters)), idxs
    
     
def run_inference(loader, params, model, device, output_spec, output_gen, output_fam):

    with tqdm(total=len(loader), unit="batch") as prog:
        
        for i, ret in enumerate(loader):

            if params.params.dataset == 'satellite_rasters_point':
                (_,_,_, batch, rasters, idx) = ret
                batch = batch.to(device)
                rasters = rasters.to(device)
                (specs, gens, fams) = model(batch.float(), rasters.float()) 
                
            else:
                (_,_,_, batch, idx) = ret
                batch = batch.to(device)
                (specs, gens, fams) = model(batch.float()) 

            output_spec[idx] = specs.detach().cpu().numpy()
            output_gen[idx] = gens.detach().cpu().numpy()
            output_fam[idx] = fams.detach().cpu().numpy()
            prog.update(1) 
    return output_spec, output_gen, output_fam

        
if __name__ == "__main__":
    np.testing.suppress_warnings()
    
    
    
    args = ['load_from_config', 'base_dir', 'device', 'test_or_train', 'num_species', 'processes', 'epoch', 'which_taxa', 'batch_size']
    ARGS = config.parse_known_args(args)       
    config.setup_main_dirs(ARGS.base_dir)
    print("number of devices visible: {dev}".format(dev=torch.cuda.device_count()))
    device = torch.device("cuda:{dev}".format(dev=ARGS.device) if ARGS.device  >= 0 else "cpu")
    print('using device: {device}'.format(device=device))
    print('load from config ', ARGS.load_from_config)
    if ARGS.load_from_config is not None:
        params = config.Run_Params(ARGS.base_dir, ARGS)
    else:
        raise FileNotFoundError("No config specified to run inference on!")

        # TODO: also add if MaxEnt or Random Forest here???
    eval_model_raw(params, ARGS.base_dir, device, ARGS.test_or_train, ARGS.which_taxa, ARGS.num_species, ARGS.processes, ARGS.epoch)
