import deepbiosphere.scripts.GEOCLEF_Utils as utils
from deepbiosphere.scripts.GEOCLEF_Utils import torch_intersection, recall_per_example, accuracy_per_example, precision_per_example
from deepbiosphere.scripts import GEOCLEF_CNN as nets
import glob
import torch
import torch.functional as F
from deepbiosphere.scripts.GEOCLEF_Config import paths, Run_Params
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
#         import pdb; pdb.set_trace()
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


# note the metrics for this species
# For each label calc the summation of true positives (tp ), true negatives (tnj), false positives (fp ), and false negatives
# true positives: classes in intersection
# false positives: classes in yhat_topk that are not in all_specs
# false negatives: classes in all_specs that are not in yhat_topk
# true negatives: classes not in yhat_topk that are not in all_specs
# 
#    yhat         y
#    1            0 FP
#    0            0 TN
#    0            1 FN
#    1            1 TP
# So I need the two X species long vectors of binary classification
# loop through those vectors, call the dict to convert to species string
# name,then add to which of the two based on value
# need a method that you call with specs, all_specs to convert to these two vectors
# then loop through the vector and do ^
# turn this into a method where I pass the dict in
# for the time being, don't wory about batching, will cross that bridge if too slow
def eval_model_new(config_name, base_dir, device='cpu', test=True, outputs='all'):
    print('setting up parameters')
    params = Run_Params(cfg_path=config_name, base_dir=base_dir)
    state = params.get_recent_model()
    dataset = run.setup_dataset(params.params.observation, params.base_dir, params.params.organism, params.params.region, params.params.normalize, params.params.no_altitude, params.params.dataset, params.params.threshold, num_species=-1)
    # TODO: doesn't handle num_species right?
    train_samp, test_samp, idxs = run.better_split_train_test(dataset)            
    train_loader = run.setup_dataloader(dataset, params.params.dataset, params.params.batch_size, 0, train_samp, params.params.model, None)
    test_loader = run.setup_dataloader(dataset, params.params.dataset, params.params.batch_size, 0, test_samp, params.params.model, None)    
    model = run.setup_model(params.params.model, dataset, params.params.pretrained, params.params.batch_norm, params.params.arch_type)
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device)    
    des = params.get_most_recent_des()
    results = Dataset.get_gbif_observations(base_dir, params.params.organism, params.params.region, params.params.observation, params.params.threshold, -1)
    # TODO: does not handle models with different style outputs ell
    inv_spec = des['inv_spec']
    inv_gen = {v: k for k, v in des['gen_dict'].items()}
    inv_fam = {v: k for k, v in des['fam_dict'].items()}
    name = None
    
    per_spec = init_perlabel_dict(inv_spec)
    per_gen = init_perlabel_dict(inv_gen)
    per_fam = init_perlabel_dict(inv_fam)
    prec = {'species' : [], 'genus': [], 'family' : []}
    rec = {'species' : [], 'genus': [], 'family' : []}
    acc = {'species' : [], 'genus': [], 'family' : []}    
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
    tick = time.time()
    # get unique species from dataset
    # one entry in per_spec for each species, sub-dict is one
    # entry per prec, rec, acc
    print(params.params.dataset)
    if params.params.dataset == 'satellite_rasters_point':
        for i, idx in enumerate(sampler):
            (specs_label, gens_label, fams_label, all_specs, all_gens, all_fams, loaded_imgs, rasters) = dataset.infer_item(idx)    
            weight = dataset.spec_freqs[specs_label]
            # unseen data performance
            # see how slow it is, if very slow can add batching
            all_specs = torch.tensor(all_specs, device=device)
            all_gens = torch.tensor(all_gens, device=device)
            all_fams = torch.tensor(all_fams, device=device)
            batch = torch.from_numpy(np.expand_dims(loaded_imgs, axis=0)).to(device)
            env_rasters = torch.from_numpy(np.expand_dims(env_rasters, axis=0)).to(device)            
            # save precision, recall, accuracy, plus raw network outputs, yhat, yhatf for each observation to the results pd dataframe
            id_ = dataset.obs[idx, Dataset.id_idx]
            (specs, gens, fams) = model(batch.float(), env_rasters) 
            prec['species'].append(precision_per_example(all_specs, specs, specs_label, dataset.spec_freqs[specs_label], device))
            rec['species'].append(recall_per_example(all_specs, specs, specs_label, dataset.spec_freqs[specs_label], device))
            acc['species'].append(accuracy_per_example(all_specs, specs, device))

            prec['genus'].append(precision_per_example(all_gens, gens, gens_label, dataset.gen_freqs[gens_label], device))
            rec['genus'].append(recall_per_example(all_gens, gens, gens_label, dataset.gen_freqs[gens_label], device))
            acc['genus'].append(accuracy_per_example(all_gens, gens, device))

            prec['family'].append(precision_per_example(all_fams, fams, fams_label, dataset.fam_freqs[fams_label], device))
            rec['family'].append(recall_per_example(all_fams, fams, fams_label, dataset.fam_freqs[fams_label], device))
            acc['family'].append(accuracy_per_example(all_fams, fams, device))    

            ys, yhats = get_fnp_vector(specs, all_specs)
            per_spec = update_lab_dict(ys, yhats, per_spec, inv_spec)
            # note number of times this species was seen total
            per_spec[inv_spec[specs_label]][1] += 1
            yg, yhatg = get_fnp_vector(gens, all_gens)
            per_gen = update_lab_dict(yg, yhatg, per_gen, inv_gen)
            per_gen[inv_gen[gens_label]][1] += 1    
            yf, yhatf = get_fnp_vector(fams, all_fams)
            per_fam = update_lab_dict(yf, yhatf, per_fam, inv_fam)    
            per_fam[inv_fam[fams_label]][1] += 1    
    else:
        for i, idx in enumerate(sampler):
            (specs_label, gens_label, fams_label, all_specs, all_gens, all_fams, loaded_imgs) = dataset.infer_item(idx)    
            weight = dataset.spec_freqs[specs_label]
            # unseen data performance
            # see how slow it is, if very slow can add batching
            all_specs = torch.tensor(all_specs, device=device)
            all_gens = torch.tensor(all_gens, device=device)
            all_fams = torch.tensor(all_fams, device=device)
            batch = torch.from_numpy(np.expand_dims(loaded_imgs, axis=0)).to(device)
            # save precision, recall, accuracy, plus raw network outputs, yhat, yhatf for each observation to the results pd dataframe
            id_ = dataset.obs[idx, Dataset.id_idx]
            if params.params.model == 'MLP_Family':
                (fams) = model(batch.float()) 
                # df.loc[df['column_name'] == some_value]
                results.loc[df['id'] == id_]['fam_precision'] = precision_per_example(all_fams, fams, fams_label, dataset.fam_freqs[fams_label], device)


    #             prec['family'].append(precision_per_example(all_fams, fams, fams_label, dataset.fam_freqs[fams_label], device))
                rec['family'].append(recall_per_example(all_fams, fams, fams_label, dataset.fam_freqs[fams_label], device))
                acc['family'].append(accuracy_per_example(all_fams, fams, device))    
                yf, yhatf = get_fnp_vector(fams, all_fams)
                per_fam = update_lab_dict(yf, yhatf, per_fam, inv_fam)    
                per_fam[inv_fam[fams_label]][1] += 1  
            elif params.params.model == 'MLP_Family_Genus':
                (fams, gens) = model(batch.float()) 
                prec['genus'].append(precision_per_example(all_gens, gens, gens_label, dataset.gen_freqs[gens_label], device))
                rec['genus'].append(recall_per_example(all_gens, gens, gens_label, dataset.gen_freqs[gens_label], device))
                acc['genus'].append(accuracy_per_example(all_gens, gens, device))

                prec['family'].append(precision_per_example(all_fams, fams, fams_label, dataset.fam_freqs[fams_label], device))
                rec['family'].append(recall_per_example(all_fams, fams, fams_label, dataset.fam_freqs[fams_label], device))
                acc['family'].append(accuracy_per_example(all_fams, fams, device))    
                # note number of times this species was seen total
                yg, yhatg = get_fnp_vector(gens, all_gens)
                per_gen = update_lab_dict(yg, yhatg, per_gen, inv_gen)
                per_gen[inv_gen[gens_label]][1] += 1    
                yf, yhatf = get_fnp_vector(fams, all_fams)
                per_fam = update_lab_dict(yf, yhatf, per_fam, inv_fam)    
                per_fam[inv_fam[fams_label]][1] += 1  
            elif params.params.model == 'SpecOnly':
                (specs) = model(batch.float()) 
                prec['species'].append(precision_per_example(all_specs, specs, specs_label, dataset.spec_freqs[specs_label], device))
                rec['species'].append(recall_per_example(all_specs, specs, specs_label, dataset.spec_freqs[specs_label], device))
                acc['species'].append(accuracy_per_example(all_specs, specs, device))
                ys, yhats = get_fnp_vector(specs, all_specs)
                per_spec = update_lab_dict(ys, yhats, per_spec, inv_spec)
                # note number of times this species was seen total
                per_spec[inv_spec[specs_label]][1] += 1
            else:
                (specs, gens, fams) = model(batch.float()) 
                prec['species'].append(precision_per_example(all_specs, specs, specs_label, dataset.spec_freqs[specs_label], device))
                rec['species'].append(recall_per_example(all_specs, specs, specs_label, dataset.spec_freqs[specs_label], device))
                acc['species'].append(accuracy_per_example(all_specs, specs, device))

                prec['genus'].append(precision_per_example(all_gens, gens, gens_label, dataset.gen_freqs[gens_label], device))
                rec['genus'].append(recall_per_example(all_gens, gens, gens_label, dataset.gen_freqs[gens_label], device))
                acc['genus'].append(accuracy_per_example(all_gens, gens, device))

                prec['family'].append(precision_per_example(all_fams, fams, fams_label, dataset.fam_freqs[fams_label], device))
                rec['family'].append(recall_per_example(all_fams, fams, fams_label, dataset.fam_freqs[fams_label], device))
                acc['family'].append(accuracy_per_example(all_fams, fams, device))    

                ys, yhats = get_fnp_vector(specs, all_specs)
                per_spec = update_lab_dict(ys, yhats, per_spec, inv_spec)
                # note number of times this species was seen total
                per_spec[inv_spec[specs_label]][1] += 1
                yg, yhatg = get_fnp_vector(gens, all_gens)
                per_gen = update_lab_dict(yg, yhatg, per_gen, inv_gen)
                per_gen[inv_gen[gens_label]][1] += 1    
                yf, yhatf = get_fnp_vector(fams, all_fams)
                per_fam = update_lab_dict(yf, yhatf, per_fam, inv_fam)    
                per_fam[inv_fam[fams_label]][1] += 1    
    tock = time.time()
    print("took ", ((tock-tick)/60), " minutes for device ", device)
    
    
    # save to disk
    filename = config_name.split('.json')[0] + name
    # make sure it's a unique file name    
    other_versions = glob.glob(base_dir + '' + filename + '.pkl')
    if len(other_versions) > 0:
        filename = filename + (len(other_versions) + 1)
    filename = base_dir + '' + filename + '.pkl'
    print('saving to ', filename)
    tosave = {
            'per_spec' : per_spec,
            'per_gen' :  per_gen,
            'per_fam' :  per_fam,
            'prec' :  prec,
            'rec' :  rec,
            'acc' :  acc
    }
    with open(filename, 'wb') as f:
        pickle.dump(tosave, f,)






# note the metrics for this species
# For each label calc the summation of true positives (tp ), true negatives (tnj), false positives (fp ), and false negatives
# true positives: classes in intersection
# false positives: classes in yhat_topk that are not in all_specs
# false negatives: classes in all_specs that are not in yhat_topk
# true negatives: classes not in yhat_topk that are not in all_specs
# 
#    yhat         y
#    1            0 FP
#    0            0 TN
#    0            1 FN
#    1            1 TP
# So I need the two X species long vectors of binary classification
# loop through those vectors, call the dict to convert to species string
# name,then add to which of the two based on value
# need a method that you call with specs, all_specs to convert to these two vectors
# then loop through the vector and do ^
# turn this into a method where I pass the dict in
# for the time being, don't wory about batching, will cross that bridge if too slow
def eval_model(config_name, base_dir, device='cpu', test=True, outputs='all'):
    print('setting up parameters')
    params = Run_Params(cfg_path=config_name, base_dir=paths.AZURE_DIR)
    state = params.get_recent_model()
    dataset = run.setup_dataset(params.params.observation, params.base_dir, params.params.organism, params.params.region, params.params.normalize, params.params.no_altitude, params.params.dataset, params.params.threshold)
    train_samp, test_samp, idxs = run.better_split_train_test(dataset, .1)            
    train_loader = run.setup_dataloader(dataset, params.params.dataset, params.params.batch_size, 0, train_samp, params.params.model, None)
    test_loader = run.setup_dataloader(dataset, params.params.dataset, params.params.batch_size, 0, test_samp, params.params.model, None)    
    model = run.setup_model(params.params.model, dataset)
    model.load_state_dict(state['model_state_dict'])
    model = model.to(device)    
    des = params.get_most_recent_des()
    # TODO: does not handle models with different style outputs ell
    inv_spec = des['inv_spec']
    inv_gen = {v: k for k, v in des['gen_dict'].items()}
    inv_fam = {v: k for k, v in des['fam_dict'].items()}
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
    tick = time.time()
    # get unique species from dataset
    # one entry in per_spec for each species, sub-dict is one
    # entry per prec, rec, acc
    per_spec = init_perlabel_dict(inv_spec)
    per_gen = init_perlabel_dict(inv_gen)
    per_fam = init_perlabel_dict(inv_fam)
    prec = {'species' : [], 'genus': [], 'family' : []}
    rec = {'species' : [], 'genus': [], 'family' : []}
    acc = {'species' : [], 'genus': [], 'family' : []}
    for i, idx in enumerate(sampler):
        (specs_label, gens_label, fams_label, all_specs, all_gens, all_fams, loaded_imgs) = dataset.infer_item(idx)    
        weight = dataset.spec_freqs[specs_label]
        # unseen data performance
        # see how slow it is, if very slow can add batching
        all_specs = torch.tensor(all_specs, device=device)
        all_gens = torch.tensor(all_gens, device=device)
        all_fams = torch.tensor(all_fams, device=device)
        batch = torch.from_numpy(np.expand_dims(loaded_imgs, axis=0)).to(device)
        if params.params.model == 'MLP_Family':
            (fams) = model(batch.float()) 
            prec['family'].append(precision_per_example(all_fams, fams, fams_label, dataset.fam_freqs[fams_label], device))
            rec['family'].append(recall_per_example(all_fams, fams, fams_label, dataset.fam_freqs[fams_label], device))
            acc['family'].append(accuracy_per_example(all_fams, fams, device))    
            yf, yhatf = get_fnp_vector(fams, all_fams)
            per_fam = update_lab_dict(yf, yhatf, per_fam, inv_fam)    
            per_fam[inv_fam[fams_label]][1] += 1  
        elif params.params.model == 'MLP_Family_Genus':
            (fams, gens) = model(batch.float()) 
            prec['genus'].append(precision_per_example(all_gens, gens, gens_label, dataset.gen_freqs[gens_label], device))
            rec['genus'].append(recall_per_example(all_gens, gens, gens_label, dataset.gen_freqs[gens_label], device))
            acc['genus'].append(accuracy_per_example(all_gens, gens, device))

            prec['family'].append(precision_per_example(all_fams, fams, fams_label, dataset.fam_freqs[fams_label], device))
            rec['family'].append(recall_per_example(all_fams, fams, fams_label, dataset.fam_freqs[fams_label], device))
            acc['family'].append(accuracy_per_example(all_fams, fams, device))    
            # note number of times this species was seen total
            yg, yhatg = get_fnp_vector(gens, all_gens)
            per_gen = update_lab_dict(yg, yhatg, per_gen, inv_gen)
            per_gen[inv_gen[gens_label]][1] += 1    
            yf, yhatf = get_fnp_vector(fams, all_fams)
            per_fam = update_lab_dict(yf, yhatf, per_fam, inv_fam)    
            per_fam[inv_fam[fams_label]][1] += 1  
        elif params.params.model == 'SpecOnly':
            (specs) = model(batch.float()) 
            prec['species'].append(precision_per_example(all_specs, specs, specs_label, dataset.spec_freqs[specs_label], device))
            rec['species'].append(recall_per_example(all_specs, specs, specs_label, dataset.spec_freqs[specs_label], device))
            acc['species'].append(accuracy_per_example(all_specs, specs, device))
            ys, yhats = get_fnp_vector(specs, all_specs)
            per_spec = update_lab_dict(ys, yhats, per_spec, inv_spec)
            # note number of times this species was seen total
            per_spec[inv_spec[specs_label]][1] += 1
        else:
            (specs, gens, fams) = model(batch.float()) 
            prec['species'].append(precision_per_example(all_specs, specs, specs_label, dataset.spec_freqs[specs_label], device))
            rec['species'].append(recall_per_example(all_specs, specs, specs_label, dataset.spec_freqs[specs_label], device))
            acc['species'].append(accuracy_per_example(all_specs, specs, device))

            prec['genus'].append(precision_per_example(all_gens, gens, gens_label, dataset.gen_freqs[gens_label], device))
            rec['genus'].append(recall_per_example(all_gens, gens, gens_label, dataset.gen_freqs[gens_label], device))
            acc['genus'].append(accuracy_per_example(all_gens, gens, device))

            prec['family'].append(precision_per_example(all_fams, fams, fams_label, dataset.fam_freqs[fams_label], device))
            rec['family'].append(recall_per_example(all_fams, fams, fams_label, dataset.fam_freqs[fams_label], device))
            acc['family'].append(accuracy_per_example(all_fams, fams, device))    

            ys, yhats = get_fnp_vector(specs, all_specs)
            per_spec = update_lab_dict(ys, yhats, per_spec, inv_spec)
            # note number of times this species was seen total
            per_spec[inv_spec[specs_label]][1] += 1
            yg, yhatg = get_fnp_vector(gens, all_gens)
            per_gen = update_lab_dict(yg, yhatg, per_gen, inv_gen)
            per_gen[inv_gen[gens_label]][1] += 1    
            yf, yhatf = get_fnp_vector(fams, all_fams)
            per_fam = update_lab_dict(yf, yhatf, per_fam, inv_fam)    
            per_fam[inv_fam[fams_label]][1] += 1    
    tock = time.time()
    print("took ", ((tock-tick)/60), " minutes for device ", device)
    
    
    # save to disk
    filename = config_name.split('.json')[0] + name
    # make sure it's a unique file name    
    other_versions = glob.glob(base_dir + '' + filename + '.pkl')
    if len(other_versions) > 0:
        filename = filename + (len(other_versions) + 1)
    filename = base_dir + '' + filename + '.pkl'
    print('saving to ', filename)
    tosave = {
            'per_spec' : per_spec,
            'per_gen' :  per_gen,
            'per_fam' :  per_fam,
            'prec' :  prec,
            'rec' :  rec,
            'acc' :  acc
    }
    with open(filename, 'wb') as f:
        pickle.dump(tosave, f,)
