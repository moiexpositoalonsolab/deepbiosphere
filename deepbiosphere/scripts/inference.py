import deepbiosphere.scripts.GEOCLEF_Utils as utils
from deepbiosphere.scripts import GEOCLEF_CNN as nets
import torch
from deepbiosphere.scripts.GEOCLEF_Config import paths, Run_Params
from deepbiosphere.scripts.GEOCLEF_Run import  setup_train_dataset, setup_model, setup_loss
import deepbiosphere.scripts.GEOCLEF_Dataset as dataset
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle


def recall_per_example(lab, guess, actual):
    # recall
    maxk = len(lab)
    pred, idxs = torch.topk(guess, maxk)
#     import pdb; pdb.set_trace()
    eq = len(list(set(idxs.tolist()[0]) & set(lab)))
    recall = eq / maxk
    top1_recall = len(list(set(idxs.tolist()[0]) & set([actual])))
    return recall, top1_recall

def recall_per_example_classes(lab, guess, actual):
    """ calculates recall per example. Returns multi-label recall + top 1 recall + all correctly predicted classes"""
    # recall
    maxk = len(lab)
    pred, idxs = torch.topk(guess, maxk)
    corr_id = list(set(idxs.tolist()[0]) & set(lab)) 
    eq = len(corr_id)
    recall = eq / maxk
    top1_recall = len(list(set(idxs.tolist()[0]) & set([actual])))
    return recall, top1_recall, corr_id

#TODO: use a threshold to determine which values to cut off the prediction at??
# this thresholding is terrible, need to fix....
# very stochastic depending on number of species in the image
# sum this across all examples and divide by num samples
def precision_per_example(lab, guess, thres=.1):
    maxk = len(lab)
    pred, idxs = torch.topk(guess, maxk)
    overlap = np.intersect1d(idxs.numpy().squeeze(), np.array(lab))
    sig = torch.nn.Softmax()
    probs = sig(guess)
    prob, idxs = torch.topk(probs, probs.shape[-1])
    cutoff = 1
    for i in range(probs[0].shape[0]):
        if probs[0][:i].sum() > thres:
            cutoff = i
            break
    return len(overlap)/ cutoff

def eval_model(config_name, base_dir, toy_dataset, epoch=None):
    params = Run_Params(cfg_path=config_name, base_dir=base_dir)
    state = params.get_recent_model()
#     state = torch.load(model_path, map_location="cpu")
    # setup_train_dataset(observation, base_dir, organism, region, normalize, altitude, dataset)
    dataset = setup_train_dataset(params.params.observation, params.base_dir, params.params.organism, params.params.region, params.params.normalize, params.params.no_altitude, params.params.dataset)
    model = setup_model(params.params.model, dataset)
    model.load_state_dict(state['model_state_dict'])
    # model = model.to(device)
    train, test = params.get_split()
    if toy_dataset:
        test = test[:1000]
    # train = splits['train']
    # test = splits['test']
    # unseen data performance
    # see how slow it is, if very slow can add batching
    # device = torch.device("cuda:0")
    #tot_prec_spc = []
    #tot_prec_gen = []
    #tot_prec_fam = []
    tracker = {sp : [0,0] for sp in dataset.spec_dict.keys()}
    tot_rec_spc =  []
    tot_rec_gen =  []
    tot_rec_fam =  []

    tot_rec1_spc =  []
    tot_rec1_gen =  []
    tot_rec1_fam =  []

    tick = time.time()
    for i in test:
        ret = dataset.infer_item(i)
        (specs_label, gens_label, fams_label, all_spec, all_gen, all_fam, batch) = ret
        batch = torch.tensor(batch).unsqueeze(axis=0)#.to(device)
        (specs, gens, fams) = model(batch.float()) 
        # put accuracy checks here!
        #prec_spc = precision_per_example(all_spec, specs, thres=.1)
       # prec_gen =precision_per_example(all_gen, gens, thres=.1)
       # prec_fam =precision_per_example(all_fam, fams, thres=.1)
        rec_spc = utils.recall_per_example_classes(all_spec, specs, specs_label)        
#         rec_spc = recall_per_example(all_spec, specs, specs_label)
#         import pdb; pdb.set_trace()
        rec_gen = recall_per_example(all_gen, gens, gens_label)
        rec_fam = recall_per_example(all_fam, fams, fams_label)
        #tot_prec_spc.append(prec_spc) 
        #tot_prec_gen.append(prec_gen) 
        #tot_prec_fam.append(prec_fam) 
        # note if prediction correct for given species
        sp = dataset.inv_spec[specs_label]
        tracker[sp][0] += rec_spc[1]
        # note given species seen
        tracker[sp][1] += 1
        tot_rec_spc.append(rec_spc[0])
        tot_rec_gen.append(rec_gen[0])
        tot_rec_fam.append(rec_fam[0])

        tot_rec1_spc.append(rec_spc[1]) 
        tot_rec1_gen.append(rec_gen[1]) 
        tot_rec1_fam.append(rec_fam[1]) 


    tock = time.time()
    diff = tock - tick
    md = diff / 60
    print("time to run is {} and {} mins".format(diff, md))
    #print(sum(tot_prec_spc)/len(tot_prec_spc), sum(tot_prec_gen)/len(tot_prec_gen), sum(tot_prec_fam)/len(tot_prec_fam))
    print(sum(tot_rec_spc)/len(tot_rec_spc), sum(tot_rec_gen)/len(tot_rec_gen), sum(tot_rec_fam)/len(tot_rec_fam))
    print(sum(tot_rec1_spc)/len(tot_rec1_spc), sum(tot_rec1_gen)/len(tot_rec1_gen), sum(tot_rec1_fam)/len(tot_rec1_fam))
    filer = config_name.split('/')[-1].split('.json')[0]
    e = '' if epoch is None else str(epoch)
    filer = base_dir + 'inference/' + filer + e +'.pkl'
    print(filer)
    tosave = {
     #   'tot_prec_spc' : tot_prec_spc,
     #   'tot_prec_gen' : tot_prec_gen,
     #   'tot_prec_fam' : tot_prec_fam,
        'tot_rec_spc' : tot_rec_spc,
        'tot_rec_gen' : tot_rec_gen,
        'tot_rec_fam' : tot_rec_fam,
        'tracker': tracker,
        'tot_rec1_spc' : tot_rec1_spc,
        'tot_rec1_gen' : tot_rec1_gen,
        'tot_rec1_fam' : tot_rec1_fam,
    }
    with open(filer, 'wb') as f:
        pickle.dump(tosave, f,)
#     return tot_prec_spc, tot_prec_gen, tot_prec_fam, tot_rec_spc, tot_rec_gen, tot_rec_fam, tot_rec1_spc, tot_rec1_gen, tot_rec1_fam, 

# config_names = [
#     # joint single
#     #flatnet + skipnet + mlp

#                  'joint_multiple_plant_cali_FlatNet_all_satellite_only_joint_mul_flatimg.json'
#                 , 'joint_multiple_plant_cali_FlatNet_all_rasters_image_joint_mul_flatras.json' # running    
#                 , 'joint_multiple_plant_cali_SkipFullFamNet_all_satellite_rasters_sheet_joint_multiple_goodras.json' # running           
#                 , 'joint_multiple_plant_cali_MLP_Family_Genus_fam_gen_rasters_point_joint_mul_mlp_famgen.json'
#                 , 'joint_multiple_plant_cali_MLP_Family_Genus_Species_all_rasters_point_joint_mul_mlp_famgenspec.json'
#                 , 'joint_multiple_plant_cali_MLP_Family_just_fam_rasters_point_joint_mul_mlp_fam.json'
    
# #         , 'joint_single_plant_cali_FlatNet_all_rasters_image_joint_sing_flat_ras.json'
# #                , 'joint_single_plant_cali_FlatNet_all_satellite_only_joint_sing_flat_img.json'

    
# #                 , 'joint_single_plant_cali_SkipFullFamNet_all_satellite_only_joint_single_img.json'    
    
# #                , 'joint_single_plant_cali_MLP_Family_Genus_fam_gen_rasters_point_joint_sing_mlp_famgen.json'
# #                , 'joint_single_plant_cali_MLP_Family_Genus_Species_all_rasters_point_joint_sing_mlp_famgenspec.json'
# #                , 'joint_single_plant_cali_MLP_Family_just_fam_rasters_point_joint_sing_mlp_fam.json'    

# ]

# tomorrow_configs = [
#                 'joint_multiple_plant_cali_SkipFullFamNet_all_satellite_rasters_sheet_joint_multiple_goodras.json' # running   
#                 ,'joint_single_plant_cali_SkipFullFamNet_all_rasters_image_joint_single_ras.json'# running
#                 , 'joint_multiple_plant_cali_FlatNet_all_rasters_image_joint_mul_flatras.json' # running
#                 , 'joint_multiple_plant_cali_SkipFullFamNet_all_satellite_only_joint_multiple_goodras.json'
#                 , 'joint_multiple_plant_cali_SkipFullFamNet_all_satellite_rasters_image_joint_multiple_badras.json'
#     # queued but may not get to it
#                , 'joint_single_plant_cali_SkipFullFamNet_all_satellite_rasters_image_joint_single_badras.json'
#                , 'joint_single_plant_cali_SkipFullFamNet_all_satellite_rasters_sheet_joint_single_goodras.json'    
#                 # GeoCLEF/nets/joint_multiple/plant/cali/SkipFullFamNet/all/satellite_only    
#                 # GeoCLEF/nets/joint_multiple/plant/cali/SkipFullFamNet/all/rasters_image 
#                 # GeoCLEF/nets/joint_multiple/plant/cali/SkipFullFamNet/all/satellite_rasters_image
# ]
# grab_old_version = [
    
# # GeoCLEF/nets/joint_multiple/plant/cali/SkipFullFamNet/all/satellite_only    
# # GeoCLEF/nets/joint_multiple/plant/cali/SkipFullFamNet/all/rasters_image 
# # GeoCLEF/nets/joint_multiple/plant/cali/SkipFullFamNet/all/satellite_rasters_image    

# ]


# for config in config_names:
#     eval_model(config)
