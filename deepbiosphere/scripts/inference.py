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

def get_alpha_diversity(base_dir, config_name, device='cpu'):
    params = Run_Params(cfg_path=config_name, base_dir=base_dir)
    state = params.get_recent_model()
    des = params.get_most_recent_des()
    
    if 'top10' in config_name:
        topk = 10
    else:
        topk = -1
    dataset = run.setup_dataset(params.params.observation, params.base_dir, params.params.organism, params.params.region, params.params.normalize, params.params.no_altitude, params.params.dataset, params.params.threshold, topk)
    train_samp, test_samp, idxs = run.better_split_train_test(dataset)            
    train_loader = run.setup_dataloader(dataset, params.params.dataset, params.params.batch_size, 0, train_samp, params.params.model, None)
    test_loader = run.setup_dataloader(dataset, params.params.dataset, params.params.batch_size, 0, test_samp, params.params.model, None)    
    model = run.setup_model(params.params.model, dataset, params.params.pretrained, params.params.batch_norm, params.params.arch_type)
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device)    
    results = Dataset.get_gbif_observations(base_dir, params.params.organism, params.params.region, params.params.observation, params.params.threshold, -1)

    results["spc_richness"] = None
    results["gen_richness"] = None
    results["fam_richness"] = None
    
    sampler = train_loader.sampler
    dataset = train_loader.dataset        
    tick = time.time()
    print('start infering')
    for i, idx in enumerate(sampler):
        if params.params.dataset == 'satellite_rasters_point':
            (specs_label, gens_label, fams_label, all_specs, all_gens, all_fams, loaded_imgs, rasters) = dataset.infer_item(idx)    
            batch = torch.from_numpy(np.expand_dims(loaded_imgs, axis=0)).to(device)
            env_rasters = torch.from_numpy(np.expand_dims(env_rasters, axis=0)).to(device)            
            (specs, gens, fams) = model(batch.float(), env_rasters) 
        else:
            (specs_label, gens_label, fams_label, all_specs, all_gens, all_fams, loaded_imgs) = dataset.infer_item(idx)    
            # unseen data performance
            # see how slow it is, if very slow can add batching
            batch = torch.from_numpy(np.expand_dims(loaded_imgs, axis=0)).to(device)
            (specs, gens, fams) = model(batch.float()) 
        
        all_specs = torch.tensor(all_specs, device=device)
        all_gens = torch.tensor(all_gens, device=device)
        all_fams = torch.tensor(all_fams, device=device)
        
        _, spec_guess = get_fnp_vector(specs, all_specs)
#         print(spec_guess)

        results.at[idx, "spc_richness"] = sum(spec_guess[0])
        
        _, gen_guess = get_fnp_vector(gens, all_gens)
        results.at[idx, "gen_richness"] = sum(gen_guess[0])
        
        _, fam_guess = get_fnp_vector(fams, all_fams)
        results.at[idx, "fam_richness"] = sum(fam_guess[0])
                                   
    sampler = test_loader.sampler
    dataset = test_loader.dataset        
    for i, idx in enumerate(sampler):
        (specs_label, gens_label, fams_label, all_specs, all_gens, all_fams, loaded_imgs) = dataset.infer_item(idx)    
        # unseen data performance
        # see how slow it is, if very slow can add batching
        batch = torch.from_numpy(np.expand_dims(loaded_imgs, axis=0)).to(device)

        (specs, gens, fams) = model(batch.float()) 
        all_specs = torch.tensor(all_specs, device=device)
        all_gens = torch.tensor(all_gens, device=device)
        all_fams = torch.tensor(all_fams, device=device)
        
        _, spec_guess = get_fnp_vector(specs, all_specs)
        results.at[idx, "spc_richness"] = sum(spec_guess[0])
        
        _, gen_guess = get_fnp_vector(gens, all_gens)
        results.at[idx, "gen_richness"] = sum(gen_guess[0])
        
        _, fam_guess = get_fnp_vector(fams, all_fams)
        results.at[idx, "fam_richness"] = sum(fam_guess[0])                                   
    tock = time.time()
    print("took ", ((tock-tick)/60), " minutes for device ", device)
    filename = config_name.split('.json')[0] + 'richness'
    # make sure it's a unique file name    
    other_versions = glob.glob(base_dir + '' + filename + '.csv')
    if len(other_versions) > 0:
        filename = filename + (len(other_versions) + 1)
    filename = base_dir + 'inference/' + filename + '.csv'
    results.to_csv(filename)                               
                                   
                                   
                                   

def pandas_inference(base_dir, config_name, test=True, device='cpu'):
    print(config_name)
    params = Run_Params(cfg_path=config_name, base_dir=base_dir)
    if config_name == 'joint_multiple_plant_cali_FlatNet_AsymmetricLossOptimized_satellite_only_top10_satonly_lowlr.json':
        state = params.get_recent_model(epoch=8)
        des = params.get_most_recent_des(epoch=8)
    else:
        state = params.get_recent_model()
        des = params.get_most_recent_des()
    
    if 'top10' in config_name:
        topk = 10
    else:
        topk = -1
#     except AttributeError: topk = -1    
    
#     print("topk: ", topk)
#     print(vars(params))
    dataset = run.setup_dataset(params.params.observation, params.base_dir, params.params.organism, params.params.region, params.params.normalize, params.params.no_altitude, params.params.dataset, params.params.threshold, topk)
    train_samp, test_samp, idxs = run.better_split_train_test(dataset, .1)            
    train_loader = run.setup_dataloader(dataset, params.params.dataset, params.params.batch_size, 0, train_samp, params.params.model, None)
    test_loader = run.setup_dataloader(dataset, params.params.dataset, params.params.batch_size, 0, test_samp, params.params.model, None)    
    model = run.setup_model(params.params.model, dataset)
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device)    
    results = Dataset.get_gbif_observations(base_dir, params.params.organism, params.params.region, params.params.observation, params.params.threshold, topk)
    name = None
    if test:
        print('inferring on test set')
        sampler = test_loader.sampler
        dataset = test_loader.dataset        
        name = 'test'
    else:
        print('inferring on train set')        
        sampler = train_loader.sampler
        dataset = train_loader.dataset        
        name = 'train'
    results = make_new_cols('fam', results)
    results = make_new_cols('gen', results)
    results = make_new_cols('spc', results)

    tick = time.time()
    for i, idx in enumerate(sampler):
        (specs_label, gens_label, fams_label, all_specs, all_gens, all_fams, loaded_imgs) = dataset.infer_item(idx)    
        weight = dataset.spec_freqs[specs_label]
        # unseen data performance
        # see how slow it is, if very slow can add batching
        batch = torch.from_numpy(np.expand_dims(loaded_imgs, axis=0)).to(device)
        # save precision, recall, accuracy, plus raw network outputs, yhat, yhatf for each observation to the results pd dataframe
        id_ = dataset.obs[idx, Dataset.id_idx]
        if params.params.model == 'MLP_Family':
            (fams) = model(batch.float())
            all_fams = torch.tensor(all_fams, device=device)
            results = run_inf_on_taxon(results, 'fam', idx, all_fams, fams, fams_label, dataset.fam_freqs[fams_label], device)
        elif params.params.model == 'MLP_Family_Genus':
            (fams, gens) = model(batch.float()) 
            all_gens = torch.tensor(all_gens, device=device)
            all_fams = torch.tensor(all_fams, device=device)        
            results = run_inf_on_taxon(results, 'fam', idx, all_fams, fams, fams_label, dataset.fam_freqs[fams_label], device)
            results = run_inf_on_taxon(results, 'gen', idx, all_gens, gens, gens_label, dataset.gen_freqs[gens_label], device)
        elif params.params.model == 'SpecOnly':
            (specs) = model(batch.float()) 
            all_specs = torch.tensor(all_specs, device=device)        
            results = run_inf_on_taxon(results, 'spc', idx, all_specs, specs, specs_label, dataset.spec_freqs[specs_label], device)
        else:

            (specs, gens, fams) = model(batch.float()) 
            all_specs = torch.tensor(all_specs, device=device)
            all_gens = torch.tensor(all_gens, device=device)
            all_fams = torch.tensor(all_fams, device=device)        
            results = run_inf_on_taxon(results, 'fam', idx, all_fams, fams, fams_label, dataset.fam_freqs[fams_label], device)
            results = run_inf_on_taxon(results, 'gen', idx, all_gens, gens, gens_label, dataset.gen_freqs[gens_label], device)
            results = run_inf_on_taxon(results, 'spc', idx, all_specs, specs, specs_label, dataset.spec_freqs[specs_label], device)
    tock = time.time()
    print("took ", ((tock-tick)/60), " minutes for device ", device)
    filename = config_name.split('.json')[0] + name
    # make sure it's a unique file name    
    other_versions = glob.glob(base_dir + '' + filename + '.csv')
    if len(other_versions) > 0:
        filename = filename + (len(other_versions) + 1)
    filename = base_dir + 'inference/' + filename + '.csv'
    results.to_csv(filename)
    
# prep the dataset for working with this data
        # save precision, recall, accuracy, plus raw network outputs, yhat, yhatf for each observation to the results pd dataframe
    # recall, top1_rec.tolist(), top1_rec_weight, intersection.tolist()
def make_new_cols(taxon, results):
    results["{}_precision_avg".format(taxon)] = None
    results["{}_precision_top1".format(taxon)] = None
    results["{}_precision_top1weight".format(taxon)] = None

    results["{}_recall_avg".format(taxon)] = None
    results["{}_recall_top1".format(taxon)] = None
    results["{}_recall_top1weight".format(taxon)] = None

    results["{}_accuracy".format(taxon)] = None
    results["{}_intersection".format(taxon)] = None
    results["{}_union".format(taxon)] = None

    results["{}_outputs".format(taxon)] = None
    results["{}_yhat".format(taxon)] = None
    results["{}_ytrue".format(taxon)] = None
    return results


def run_inf_on_taxon(results, taxon, idx, labels, guess, toplabel, weight, device):
    results.at[idx, "{}_precision_avg".format(taxon)] = precision_per_example(labels, guess, toplabel, weight, device)[0]
    results.at[idx, "{}_precision_top1".format(taxon)] = precision_per_example(labels, guess, toplabel, weight, device)[1]
    results.at[idx, "{}_precision_top1weight".format(taxon)] = precision_per_example(labels, guess, toplabel, weight, device)[2]
    results.at[idx, "{}_recall_avg".format(taxon)] = recall_per_example(labels, guess, toplabel, weight, device)[0]
    results.at[idx, "{}_recall_top1".format(taxon)] = recall_per_example(labels, guess, toplabel, weight, device)[1]
    results.at[idx, "{}_recall_top1weight".format(taxon)] = recall_per_example(labels, guess, toplabel, weight, device)[2]
    results.at[idx, "{}_accuracy".format(taxon)] = accuracy_per_example(labels, guess, device)[0]
    results.at[idx, "{}_intersection".format(taxon)] = accuracy_per_example(labels, guess, device)[1]
    results.at[idx, "{}_union".format(taxon)] = accuracy_per_example(labels, guess, device)[2]
    results.at[idx, "{}_outputs".format(taxon)] = guess.tolist()
    results.at[idx, "{}_yhat".format(taxon)] = get_fnp_vector(guess, labels)[1]
    results.at[idx, "{}_ytrue".format(taxon)] = get_fnp_vector(guess, labels)[0]
    return results



def get_fnp_vector(guess, lab):
    guess = torch.sigmoid(guess)
    yhat_size = (guess > .5).sum()
    pred, idxs = torch.topk(guess, yhat_size)
    # make sure when return to take off gpu for speed
    # dict ops are faster on cpu 
    yhat = torch.zeros_like(guess, dtype=torch.uint8, device='cpu')
    y = torch.zeros_like(guess, dtype=torch.uint8, device='cpu')
    yhat[:,idxs] +=1
    y[:,lab] += 1
    return y.tolist(), yhat.tolist()


def init_perlabel_dict(inv_dict):
    return {
    id_ : [{
        'tp' : 0,
        'fp' : 0,
        'tn' : 0,
        'fn' : 0
    },0]
    for id_ in inv_dict.values()}

def update_lab_dict(y, yhat, label_dict, inv_dict):
    
    for i, (maybe, corr) in enumerate(zip(yhat[0], y[0])):
        if maybe and corr:
            label_dict[inv_dict[i]][0]['tp'] += 1
        if maybe and not corr:
            label_dict[inv_dict[i]][0]['fp'] += 1
        if not maybe and not corr:
            label_dict[inv_dict[i]][0]['tn'] += 1   
        if not maybe and corr:
            label_dict[inv_dict[i]][0]['fn'] += 1       
    return label_dict


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
    df_spec = pd.DataFrame(output_spec)
    df_gen = pd.DataFrame(output_gen)
    df_fam = pd.DataFrame(output_fam)    
    inv_gen = {v: k for k, v in dset.gen_dict.items()}
    inv_fam = {v: k for k, v in dset.fam_dict.items()}
    df_spec.columns = [dset.inv_spec[i] for i in range(dset.num_specs)]
    df_gen.columns = [inv_gen[i] for i in range(dset.num_gens)]
    df_fam.columns = [inv_fam[i] for i in range(dset.num_fams)]
    to_transfer = ['lat', 'lon', 'region', 'city', 'NA_L3NAME', 'US_L3NAME', 'NA_L2NAME', 'NA_L1NAME', 'test']
    df_spec[to_transfer] = obs[to_transfer]
    df_gen[to_transfer] = obs[to_transfer]
    df_fam[to_transfer] = obs[to_transfer]
    
    if num_species < 0:
        nsp = 'all_spec'
    else:
        nsp = "top_{}_spec".format(num_species)    
    # get good name and save to csv
    name_spec = "{}_{}_{}_{}_{}_{}.csv".format(params.get_cfg_name(), 'species' , nsp, datetime.now().day, datetime.now().month, datetime.now().year)
    name_gen = "{}_{}_{}_{}_{}_{}.csv".format(params.get_cfg_name(), 'genera' , nsp, datetime.now().day, datetime.now().month, datetime.now().year)
    name_fam = "{}_{}_{}_{}_{}_{}.csv".format(params.get_cfg_name(), 'family' , nsp, datetime.now().day, datetime.now().month, datetime.now().year)
    df_spec.to_csv( base_dir + '/inference/' + name_spec)
    df_gen.to_csv( base_dir + '/inference/' + name_gen)
    df_fam.to_csv( base_dir + '/inference/' + name_fam)    
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
