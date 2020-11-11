import copy
import pickle
import pandas as pd
import argparse
import time
import numpy as np
import socket
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import itertools
import gc
import csv
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import math
from tqdm import tqdm
from deepbiosphere.scripts import GEOCLEF_CNN as cnn
# from deepbiosphere.scripts import inference as inference
from deepbiosphere.scripts import GEOCLEF_Dataset as Dataset
from deepbiosphere.scripts import GEOCLEF_Loss as losses
from deepbiosphere.scripts import GEOCLEF_Utils as utils
from deepbiosphere.scripts import GEOCLEF_Config as config




def better_split_train_test(full_dat, split_amt):
#     shuffle = np.random.permutation(np.arange(len(dset)))
#     split = int(len(idxs)*split_amt)    
#     test = set()
#     total = len(dset)
#     i = 0
#     while len(test) <= split:
#         test.update(full_dat.obs[shuffle[i], dataset.ids_idx])
#         i += 1    
#     test_idx = []
#     train_idx = []
#     for idx in np.arange(len(full_dat)):
#         test_idx.append(idx) if full_dat.obs[idx,0] in test else train_idx.append(idx)
#     train_sampler = SubsetRandomSampler(train_idx)
#     valid_sampler = SubsetRandomSampler(test_idx)
    test_idx = full_dat.test
    train_idx, full_dat.train
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(test_idx)
    return train_sampler, valid_sampler, {'train': train_idx, 'test' : test_idx}


def old_split_train_test(full_dat, split_amt):
    '''grab split_amt% of labeled data for holdout testing'''
    idxs = np.random.permutation(len(full_dat))
    split = int(len(idxs)*split_amt)
    training_idx, test_idx = idxs[:split], idxs[split:]
    train_sampler = SubsetRandomSampler(training_idx)
    valid_sampler = SubsetRandomSampler(test_idx)
    return train_sampler, valid_sampler, {'train': training_idx, 'test' : test_idx}


      

def check_mem():
    '''Grabs all in-scope tensors '''
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.device)
        except: pass



def setup_dataset(observation, base_dir, organism, region, normalize, altitude, dataset, threshold):
    '''grab and setup train or test dataset'''
    
    if dataset == 'satellite_only':
        return Dataset.HighRes_Satellite_Images_Only(base_dir, organism, region, observation, altitude, threshold)
        
    elif dataset == 'satellite_rasters_image':
        return Dataset.HighRes_Satellite_Rasters_LowRes(base_dir, organism, region, normalize, observation, altitude, threshold)
    elif dataset == 'satellite_rasters_point':
        return Dataset.HighRes_Satellite_Rasters_Point(base_dir, organism, region, observation, altitude, normalize, threshold)
    elif dataset == 'rasters_image':
        return Dataset.Bioclim_Rasters_Image(base_dir, organism, region, normalize, observation, threshold)
    elif dataset == 'rasters_point':
        return Dataset.Bioclim_Rasters_Point(base_dir, organism, region, normalize, observation, threshold)
        
    elif dataset == 'satellite_rasters_sheet':
        return Dataset.HighRes_Satellite_Rasters_Sheet(base_dir, organism, region, normalize, observation, altitude, threshold)
    else: 
        raise NotImplementedError

        
def setup_model(model, train_dataset):
    
    num_specs = train_dataset.num_specs
    num_fams = train_dataset.num_fams
    num_gens = train_dataset.num_gens
    
    if model == 'SVM':
        raise NotImplementedError
    elif model == 'RandomForest':
        raise NotImplementedError
    elif model == 'SVM':
        raise NotImplementedError
    # some flavor of convnet architecture
    elif model == 'OGNoFamNet':
        return cnn.OGNoFamNet(species=num_specs, families=num_fams, genuses=num_gens, num_channels=train_dataset.channels)        
    elif model == 'OGNet':
        return cnn.OGNet(species=num_specs, families=num_fams, genuses=num_gens, num_channels=train_dataset.channels)        
    elif model == 'SkipFCNet':
        return cnn.SkipFCNet(species=num_specs, families=num_fams, genuses=num_gens, num_channels=train_dataset.channels)        
    elif model == 'SkipNet':
        return cnn.SkipNet(species=num_specs, families=num_fams, genuses=num_gens, num_channels=train_dataset.channels)        
    elif model == 'MixNet':
        return cnn.MixNet(species=num_specs, families=num_fams, genuses=num_gens, num_channels=train_dataset.channels, env_rasters=train_dataset.num_rasters)
    elif model == 'MixFullNet':
        return cnn.MixFullNet(species=num_specs, families=num_fams, genuses=num_gens, num_channels=train_dataset.channels, env_rasters=train_dataset.num_rasters)
    elif model == 'SkipFullFamNet':
        return cnn.SkipFullFamNet(species=num_specs, families=num_fams, genuses=num_gens, num_channels=train_dataset.channels)
    elif model == 'FlatNet':
        return cnn.FlatNet(species=num_specs, families=num_fams, genuses=num_gens, num_channels=train_dataset.channels)
    elif model == 'MLP_Family':
        return cnn.MLP_Family(families=num_fams, env_rasters=train_dataset.num_rasters)
    elif model == 'MLP_Family_Genus':
        return cnn.MLP_Family_Genus(families=num_fams, genuses=num_gens, env_rasters=train_dataset.num_rasters)    
    elif model == 'MLP_Family_Genus_Species':
        return cnn.MLP_Family_Genus_Species(families=num_fams, genuses=num_gens, species=num_specs, env_rasters=train_dataset.num_rasters)    
    elif model == 'SpecOnly':
        return cnn.SpecOnly(species=num_specs, num_channels=train_dataset.channels)
    else: 
        exit(1), "if you reach this, you got a real problem bucko"

        
def setup_dataloader(dataset, dtype,batch_size, processes, sampler, model):
    if dtype == 'satellite_rasters_point':
        collate_fn = joint_raster_collate_fn
    else:
        collate_fn = joint_collate_fn
    dataloader = DataLoader(dataset, batch_size, pin_memory=False, num_workers=processes, collate_fn=collate_fn, sampler=sampler)

    return dataloader


    
def setup_loss(observation, dataset, loss, unweighted, device, loss_type):

    if loss == 'none':
        return None, None, None
    
    if loss =='BrierAll':
        spec_loss= losses.BrierAll(loss_type)
        gen_loss = losses.BrierAll(loss_type)
        fam_loss = losses.BrierAll(loss_type)
    elif loss == 'BrierPresenceOnly':
        spec_loss= losses.BrierPresenceOnly(loss_type)
        gen_loss = losses.BrierPresenceOnly(loss_type)
        fam_loss = losses.BrierPresenceOnly(loss_type)    
    elif loss == 'MultiLabelMarginLoss':
        spec_loss= losses.BrierAll(loss_type)
        gen_loss = losses.BrierAll(loss_type)
        fam_loss = losses.BrierAll(loss_type)    
    elif loss == 'AsymmetricLoss':
        spec_loss= losses.AsymmetricLoss()
        gen_loss = losses.AsymmetricLoss()
        fam_loss = losses.AsymmetricLoss()        
    elif loss == 'AsymmetricLossOptimized':    
        spec_loss= losses.AsymmetricLossOptimized()
        gen_loss = losses.AsymmetricLossOptimized()
        fam_loss = losses.AsymmetricLossOptimized()
    if not unweighted:
        spec_freq = Dataset.freq_from_dict(dataset.spec_freqs)
        gen_freq = Dataset.freq_from_dict(dataset.gen_freqs)
        fam_freq = Dataset.freq_from_dict(dataset.fam_freqs)        
        spec_freq = 1.0 / torch.tensor(spec_freq, dtype=torch.float, device=device)
        gen_freq = 1.0 / torch.tensor(gen_freq, dtype=torch.float, device=device)
        fam_freq = 1.0 / torch.tensor(fam_freq, dtype=torch.float, device=device)
        if loss == 'BCEWithLogits':
            spec_loss = torch.nn.BCEWithLogitsLoss(spec_freq, reduction=loss_type)
            gen_loss = torch.nn.BCEWithLogitsLoss(gen_freq, reduction=loss_type)
            fam_loss = torch.nn.BCEWithLogitsLoss(fam_freq, reduction=loss_type)
        elif loss == 'CrossEntropyPresenceOnly':
            spec_loss= losses.CrossEntropyPresenceOnly(spec_freq, type=loss_type)
            gen_loss = losses.CrossEntropyPresenceOnly(gen_freq, type=loss_type)
            fam_loss = losses.CrossEntropyPresenceOnly(fam_freq, type=loss_type)
        else:
            raise NotImplementedError
    else:
        if loss == 'BCEWithLogits':
            spec_loss = torch.nn.BCEWithLogitsLoss(reduction=loss_type)
            gen_loss = torch.nn.BCEWithLogitsLoss(reduction=loss_type)
            fam_loss = torch.nn.BCEWithLogitsLoss(reduction=loss_type)
        elif loss == 'CrossEntropyPresenceOnly':
            spec_loss= losses.CrossEntropyPresenceOnly(torch.ones(num_specs, dtype=torch.float, device=device), reduction=loss_type)
            gen_loss = losses.CrossEntropyPresenceOnly(torch.ones(num_gens, dtype=torch.float, device=device), reduction=loss_type)
            fam_loss = losses.CrossEntropyPresenceOnly(torch.ones(num_fams, dtype=torch.float, device=device), reduction=loss_type)
        else:
            raise NotImplementedError

            
    if loss == 'just_fam':
        gen_loss = None
        spec_loss = None
    elif loss == 'fam_gen':
        spec_loss == None
    elif loss == 'spec_only':
        fam_loss = None
        gen_loss = None

        
    return spec_loss, gen_loss, fam_loss
    
    
def clean_gpu(device):
    if device is not None:
        print("cleaning gpu")            
        torch.cuda.empty_cache()
        
# def single_collate_fn(batch): 
#     # batch is a list of tuples of (composite_label <np array [3]>, images <np array [6, 256, 256]>)   
#     labs, img = zip(*batch) 
#     print(labs[0][0], labs[0], labs)
#     lbs = [torch.tensor(l[0], dtype=torch.long) for l in labs]      
#     img = [i.astype(np.uint8, copy=False) for i in img]
#     imgs = [torch.from_numpy(i) for i in img]
#     print(torch.stack(lbs).shape)
#     return torch.stack(lbs), torch.stack(imgs)  

def joint_collate_fn(batch):
    # batch is a list of tuples of (specs_label, gens_label, fams_label, images)  
    all_specs = []
    all_gens = []
    all_fams = []
    imgs = []
    num_specs = 3988 # TREMOVE@!!
    num_gens = 1243
    num_fams = 241
    #(specs_label, gens_label, fams_label, images)  
    for (spec, gen, fam, img) in batch:
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
    return torch.stack(all_specs), torch.stack(all_gens), torch.stack(all_fams), torch.from_numpy(np.stack(imgs))

def joint_raster_collate_fn(batch):
    # batch is a list of tuples of (specs_label, gens_label, fams_label, images, env_rasters)  
    all_specs = []
    all_gens = []
    all_fams = []
    imgs = []
    rasters = []
    #(specs_label, gens_label, fams_label, images, env_rasters)  
    for (spec, gen, fam, img, raster) in batch:
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
    return torch.stack(all_specs), torch.stack(all_gens), torch.stack(all_fams), torch.from_numpy(np.stack(imgs)), torch.from_numpy(np.stack(rasters))

# def joint_rasteronly_collate_fn(batch):
#     # batch is a list of tuples of (specs_label, gens_label, fams_label, images, env_rasters)  
#     all_specs = []
#     all_gens = []
#     all_fams = []
#     rasters = []
#     #(specs_label, gens_label, fams_label, images, env_rasters)  
#     for (spec, gen, fam, raster) in batch:
#         specs_tens = torch.zeros(num_specs)
#         specs_tens[spec] += 1
#         all_specs.append(specs_tens)

#         gens_tens = torch.zeros(num_gens)
#         gens_tens[gen] += 1
#         all_gens.append(gens_tens)

#         fams_tens = torch.zeros(num_fams)
#         fams_tens[fam] += 1
#         all_fams.append(fams_tens)
#         rasters.append(raster)
#     return torch.stack(all_specs), torch.stack(all_gens), torch.stack(all_fams), torch.from_numpy(np.stack(rasters))

                                                       
                                                       
                                                       
def test_batch(test_loader, tb_writer, device, net, observation, epoch, loss, model, dataset):
    if model == 'SpecOnly':
        if observation == 'single' or observation == 'single_single':
            return test_single_speconly_batch(test_loader, tb_writer, device, net, epoch)
        else:
            return test_joint_speconly_batch(test_loader, tb_writer, device, net, epoch)
    elif model == 'MLP_Family':
        if observation == 'single' or observation == 'single_single':
            return test_single_obs_fam(test_loader, tb_writer, device, net, epoch)
        else:
            return test_joint_obs_fam(test_loader, tb_writer, device, net, epoch)
    elif model == 'MLP_Family_Genus':
        if observation == 'single' or observation == 'single_single':
            return test_single_obs_rastersonly_famgen(test_loader, tb_writer, device, net, epoch)
        else:
            return test_joint_obs_rastersonly_famgen(test_loader, tb_writer, device, net, epoch)

    elif dataset == 'satellite_rasters_point':
        if observation == 'single' or observation == 'single_single':
            return test_single_obs_rasters_batch(test_loader, tb_writer, device, net, epoch)
        else:
            return test_joint_obs_rasters_batch(test_loader, tb_writer, device, net, epoch)
    else:
        if observation == 'single' or observation == 'single_single':
            return test_single_obs_batch(test_loader, tb_writer, device, net, epoch)
        else:
            return test_joint_obs_batch(test_loader, tb_writer, device, net, epoch)
                                 
                                                       
def test_single_obs_batch(test_loader, tb_writer, device, net, epoch):
     with tqdm(total=len(test_loader), unit="batch") as prog:
        all_accs = []
        all_spec = []
        all_gen = []
        all_fam = []
        for i, (specs_label, gens_label, fams_label, loaded_imgs) in enumerate(test_loader):

            specs_label = specs_label.to(device)
            gens_label = gens_label.to(device)
            fams_label = fams_label.to(device)
            batch = batch.to(device)
            (outputs, genus, family) = net(batch.float())
            spec_accs = utils.topk_acc(outputs, specs_label, topk=(30,1), device=device) # magic no from CELF2020
            gens_accs = utils.topk_acc(genus, gens_label, topk=(30,1), device=device) # magic no from CELF2020
            fam_accs = utils.topk_acc(family, fams_label, topk=(30,1), device=device) # magic no from CELF2020
            prog.set_description("top 30: {acc0}  top1: {acc1}".format(acc0=spec_accs[0], acc1=spec_accs[1]))
            all_spec.append(spec_accs)
            all_gen.append(gens_accs)
            all_fam.append(fam_accs)
            if tb_writer is not None:
                tb_writer.add_scalar("test/30_spec_accuracy", spec_accs[0], epoch)
                tb_writer.add_scalar("test/1_spec_accuracy", spec_accs[1], epoch)  

                tb_writer.add_scalar("test/30_gen_accuracy", gens_accs[0], epoch)
                tb_writer.add_scalar("test/1_gen_accuracy", gens_accs[1], epoch)  

                tb_writer.add_scalar("test/30_fam_accuracy", fam_accs[0], epoch)
                tb_writer.add_scalar("test/1_fam_accuracy", fam_accs[1], epoch)   

            prog.update(1)
        prog.close()
        return all_spec, all_gen, all_fam
 
    #TODO: fix to work with recall_per_example
def test_joint_obs_rasters_batch(test_loader, tb_writer, device, net, epoch):
    with tqdm(total=len(test_loader), unit="batch") as prog:
        allspec = []
        allgen = []
        allfam = []
        for i, (specs_label, gens_label, fams_label, imgs, env_rasters) in enumerate(test_loader):
            imgs = imgs.to(device)
            env_rasters = torch.from_numpy(env_rasters).to(device)
            (outputs, gens, fams) = net(env_rasters.float(), env_rasters.float()) 
            specaccs, totspec_accs = utils.num_corr_matches(outputs, specs_lab) # magic no from CELF2020
            genaccs, totgen_accs = utils.num_corr_matches(gens, gens_label) # magic no from CELF2020                        
            famaccs, totfam_accs = utils.num_corr_matches(fams, fams_label) # magic no from CELF2020  
            #TODO: add other accuracy metrics??
            prog.set_description("mean accuracy across batch: {acc0}".format(acc0=specaccs.mean()))
            prog.update(1)          
            if tb_writer is not None:
                tb_writer.add_scalar("test/avg_spec_accuracy", specaccs.mean(), epoch)
                tb_writer.add_scalar("test/avg_gen_accuracy", genaccs.mean(), epoch)
                tb_writer.add_scalar("test/avg_fam_accuracy", famaccs.mean(), epoch)                        
            allspec.append(totspec_accs)
            allgen.append(totgen_accs)
            allfam.append(totfam_accs)
    prog.close()
    return allfam, allgen, allspec
                                                       
                                                       
def test_single_obs_rasters_batch(test_loader, tb_writer, device, net, epoch):
    with tqdm(total=len(test_loader), unit="batch") as prog:
        allspec = []
        allgen = []
        allfam = []
        for i, (specs_label, gens_label, fams_label, imgs, env_rasters) in enumerate(test_loader):
            imgs = imgs.to(device)
            env_rasters = env_rasters.to(device)
            specs_lab = specs_label.to(device)                                     
            gens_label = gens_label.to(device)
            fams_label = fams_label.to(device)
            (outputs, gens, fams) = net(imgs.float(), env_rasters.float()) 
            spec_accs = utils.topk_acc(outputs, specs_label, topk=(30,1), device=device) # magic no from CELF2020
            gens_accs = utils.topk_acc(gens, gens_label, topk=(30,1), device=device) # magic no from CELF2020
            fam_accs = utils.topk_acc(fams, fams_label, topk=(30,1), device=device) # magic no from CELF2020
            prog.set_description("top 30: {acc0}  top1: {acc1}".format(acc0=spec_accs[0], acc1=spec_accs[1]))
            all_spec.append(spec_accs)
            all_gen.append(gens_accs)
            all_fam.append(fam_accs)
            if tb_writer is not None:
                tb_writer.add_scalar("test/30_spec_accuracy", spec_accs[0], epoch)
                tb_writer.add_scalar("test/1_spec_accuracy", spec_accs[1], epoch)  

                tb_writer.add_scalar("test/30_gen_accuracy", gens_accs[0], epoch)
                tb_writer.add_scalar("test/1_gen_accuracy", gens_accs[1], epoch)  

                tb_writer.add_scalar("test/30_fam_accuracy", fam_accs[0], epoch)
                tb_writer.add_scalar("test/1_fam_accuracy", fam_accs[1], epoch)                          
        allspec.append(totspec_accs)
        allgen.append(totgen_accs)
        allfam.append(totfam_accs)
    prog.close()
    return allfam, allgen, allspec

def test_joint_obs_fam(test_loader, tb_writer, device, net, epoch):
    means = []
    all_accs = []
    mean_accs = []
    all_frec = []    
    all_tf = []    
    sampler = test_loader.sampler
    dataset = test_loader.dataset
    with tqdm(total=len(sampler), unit="example") as prog:
        for i, idx in enumerate(sampler):
        # return (specs_label, gens_label, fams_label, all_spec, all_gen, all_fam, images)
            (_, _, fams_label, _, _, all_fams, loaded_imgs) = dataset.infer_item(idx)            
            loaded_imgs = torch.from_numpy(np.expand_dims(loaded_imgs, axis=0)).to(device)
            fams = net(loaded_imgs.float()) 
            weight = dataset.fam_freqs[fams_label]
            famrec, famtop1 = utils.recall_per_example(fams, all_fams, fams_label, weight) # magic no from CELF2020  
            all_frec.append(famrec)
            all_tf.append(famtop1)            
            prog.set_description("mean recall across batch: {acc0}".format(acc0=famrec))
            prog.update(1)          
            all_accs.append(famrec)
            mean_accs.append(famtop1)
            means.append(famrec * 100)
        if tb_writer is not None:
            tb_writer.add_scalar("test/avg_fam_recall",  mean(all_frec) * 100, epoch)
            tb_writer.add_scalar("test/avg_fam_top1_recall", mean(all_tf) * 100, epoch)            
    prog.close()
    return means, all_accs, mean_accs

def test_single_obs_fam(test_loader, tb_writer, device, net, epoch):
    with tqdm(total=len(test_loader), unit="batch") as prog:
        means = []
        all_accs = []
        mean_accs = []
        for i, (_, _, fams_label, env_rasters) in enumerate(test_loader):
            env_rasters = env_rasters.to(device)
            fams_label = fams_label.to(device)
            fams = net(env_rasters.float()) 
            fam_accs = utils.topk_acc(fams, fams_label, topk=(30,1), device=device) # magic no from CELF2020
            prog.set_description("top 30: {acc0}  top1: {acc1}".format(acc0=fam_accs[0], acc1=fam_accs[1]))
            prog.update(1)          
            if tb_writer is not None:
                tb_writer.add_scalar("test/30_fam_accuracy", fam_accs[0], epoch)
                tb_writer.add_scalar("test/1_fam_accuracy", fam_accs[1], epoch)   
            all_accs.append(fam_accs)
            mean_accs.append(fam_accs)
            means.append(fam_accs.mean())
    prog.close()
    return means, all_accs, mean_accs

def test_joint_obs_rastersonly_all(test_loader, tb_writer, device, net, epoch):
    with tqdm(total=len(test_loader), unit="batch") as prog:
        means = []
        all_accs = []
        mean_accs = []
        for i, (specs_label, gens_label, fams_label, env_rasters) in enumerate(test_loader):
            env_rasters = env_rasters.to(device)
            fams_label = fams_label.to(device)
            gens_label = gens_label.to(device)
            specs_label = specs_label.to(device)
            fams, gens, specs = net(env_rasters.float()) 
            genaccs, totgen_accs = utils.num_corr_matches(gens, gens_label) # magic no from CELF2020                                    
            famaccs, totfam_accs = utils.num_corr_matches(fams, fams_label) # magic no from CELF2020  
            specaccs, totspec_accs = utils.num_corr_matches(specs, specs_label) # magic no from CELF2020              
            #TODO: add other accuracy metrics??
            prog.set_description("mean accuracy across batch: {acc0}".format(acc0=specaccs.mean()))
            prog.update(1)          
            if tb_writer is not None:
                tb_writer.add_scalar("test/avg_fam_accuracy", famaccs.mean(), epoch)
                tb_writer.add_scalar("test/avg_gen_accuracy", genaccs.mean(), epoch)
                tb_writer.add_scalar("test/avg_spec_accuracy", specaccs.mean(), epoch)
            all_accs.append(totfam_accs)
            mean_accs.append(totgen_accs)
            means.append(totspec_accs)
    prog.close()
    return means, all_accs, mean_accs

def mean(lst): 
    return sum(lst) / len(lst) 

def test_joint_obs_rastersonly_famgen(test_loader, tb_writer, device, net, epoch):
    means = []
    all_accs = []
    mean_accs = []
    all_grec = []
    all_frec = []
    all_tg = []
    all_tf = []
    sampler = test_loader.sampler
    dataset = test_loader.dataset
    with tqdm(total=len(sampler), unit="example") as prog:
        for i, idx in enumerate(sampler):
        # return (specs_label, gens_label, fams_label, all_spec, all_gen, all_fam, images)
            (_, gens_label, fams_label, _, all_gens, all_fams, env_rasters) = dataset.infer_item(idx)            
            env_rasters= torch.from_numpy(np.expand_dims(env_rasters, axis=0)).to(device)
            fams, gens = net(env_rasters.float()) 
            fam_weight = dataset.fam_freqs[fams_label]
            gen_weight = dataset.gen_freqs[gens_label]
            famrec, famtop1 = utils.recall_per_example(fams, all_fams, fams_label, fam_weight) # magic no from CELF2020  
            genrec, gentop1 = utils.recall_per_example(gens, all_gens, gens_label, gen_weight) # magic no from CELF2020  
            all_grec.append(genrec)
            all_frec.append(famrec)
            all_tg.append(gentop1)
            all_tf.append(famtop1)
            prog.set_description("mean recall across batch: {acc0}".format(acc0=genrec))
            prog.update(1)          
            all_accs.append((famrec, genrec))
            mean_accs.append((famtop1, gentop1))
            means.append((famrec * 100, genrec * 100))
        if tb_writer is not None:
            tb_writer.add_scalar("test/avg_fam_recall", mean(all_frec) * 100, epoch)
            tb_writer.add_scalar("test/avg_fam_top1_recall", mean(all_tf) * 100, epoch) 
            tb_writer.add_scalar("test/avg_gen_recall", mean(all_grec) * 100, epoch)
            tb_writer.add_scalar("test/avg_gen_top1_recall", mean(all_tg) * 100, epoch)                             
    prog.close()
    return means, all_accs, mean_accs    
    
                                                       
def test_single_obs_rastersonly_famgen(test_loader, tb_writer, device, net, epoch):
    with tqdm(total=len(test_loader), unit="batch") as prog:
        means = []
        all_accs = []
        mean_accs = []
        for i, (_, gens_label, fams_label, env_rasters) in enumerate(test_loader):
            env_rasters = env_rasters.to(device)
            fams_label = fams_label.to(device)
            gens_label = gens_label.to(device)
            fams, gens = net(env_rasters.float()) 
            gens_accs = utils.topk_acc(gens, gens_label, topk=(30,1), device=device) # magic no from CELF2020
            fam_accs = utils.topk_acc(fams, fams_label, topk=(30,1), device=device) # magic no from CELF2020

            prog.set_description("top 30: {acc0}  top1: {acc1}".format(acc0=gens_accs[0], acc1=gens_accs[1]))

            prog.update(1)          
            if tb_writer is not None:

                tb_writer.add_scalar("test/30_gen_accuracy", gens_accs[0], epoch)
                tb_writer.add_scalar("test/1_gen_accuracy", gens_accs[1], epoch)  

                tb_writer.add_scalar("test/30_fam_accuracy", fam_accs[0], epoch)
                tb_writer.add_scalar("test/1_fam_accuracy", fam_accs[1], epoch)   
            all_accs.append(fam_accs)
            mean_accs.append(gens_accs)
            means.append(gens_accs.mean())
    prog.close()
    return means, all_accs, mean_accs

def test_joint_obs_batch(test_loader, tb_writer, device, net, epoch):
    
    allspec = []
    allgen = []
    allfam = []
    all_grec = []
    all_frec = []
    all_srec = []    
    all_tg = []
    all_tf = []    
    all_ts = []
    sampler = test_loader.sampler
    dataset = test_loader.dataset
    with tqdm(total=len(sampler), unit="example") as prog:
        for i, idx in enumerate(sampler):
            (specs_label, gens_label, fams_label, all_specs, all_gens, all_fams, loaded_imgs) = dataset.infer_item(idx)    
            # what you use for recreating labels if trianing from restart
#             ob = dataset.occs.iloc[idx]
#             sp = ob.species
#             # reconstruct true label using class mapping from model
#             all_specs = [dataset.spec_dict[s] for s in ob.all_specs_name]
#             specs_label = dataset.spec_dict[ob.species]            

            batch = torch.from_numpy(np.expand_dims(loaded_imgs, axis=0)).to(device)
            (outputs, gens, fams) = net(batch.float())
            fam_weight = dataset.fam_freqs[fams_label]
            gen_weight = dataset.gen_freqs[gens_label]
            spec_weight = dataset.spec_freqs[specs_label]
            famrec, famtop1 = utils.recall_per_example(fams, all_fams, fams_label, fam_weight) # magic no from CELF2020  
            genrec, gentop1 = utils.recall_per_example(gens, all_gens, gens_label, gen_weight) # magic no from CELF2020  
            specrec, spectop1 = utils.recall_per_example(outputs, all_specs, specs_label, spec_weight) # magic no from CELF2020
#             import pdb; pdb.set_trace()s
            
            all_grec.append(genrec)
            all_frec.append(famrec)
            all_srec.append(specrec)
            all_tg.append(gentop1)
            all_tf.append(famtop1)
            all_ts.append(spectop1)
            prog.set_description("mean recall across batch: {acc0}".format(acc0=specrec))
            prog.update(1)                               
            allspec.append((specrec, spectop1))
            allfam.append(( famrec, famtop1 ))
            allgen.append(( genrec, gentop1 ))
        if tb_writer is not None:
            tb_writer.add_scalar("test/avg_fam_recall", mean(all_frec) * 100, epoch)
            tb_writer.add_scalar("test/avg_fam_top1_recall", mean(all_tf) * 100, epoch) 
            tb_writer.add_scalar("test/avg_gen_recall", mean(all_grec) * 100, epoch)
            tb_writer.add_scalar("test/avg_gen_top1_recall", mean(all_tg) * 100, epoch)
            tb_writer.add_scalar("test/avg_spec_recall", mean(all_srec) * 100, epoch)
            tb_writer.add_scalar("test/avg_spec_top1_recall", mean(all_ts) * 100, epoch) 
            
    prog.close()
    return allfam, allgen, allspec

def test_joint_speconly_batch(test_loader, tb_writer, device, net, epoch):

    means = []
    all_accs = []
    mean_accs = []
    all_spec = []
    all_sp1 = []
    sampler = test_loader.sampler
    dataset = test_loader.dataset
    with tqdm(total=len(sampler), unit="example") as prog:
        for i, idx in enumerate(sampler):
            # specs label is top1, all_spec is all species
            (specs_label, _, _, all_spec, _, _, loaded_imgs) = dataset.infer_item(idx)
            batch = torch.from_numpy(np.expand_dims(loaded_imgs, axis=0)).to(device)
            outputs = net(batch.float()) 
            # recall, top1_recall
            spec_weight = dataset.spec_freqs[specs_label]
            specrec, spectop1 = utils.recall_per_example(outputs, all_specs, specs_label, spec_weight) # magic no from CELF2020
            prog.set_description("mean recall across batch: {acc0}".format(acc0=specrec))
            all_spec.append(specrec)
            all_sp1.append(spectop1)
            prog.update(1)          
            all_accs.append(specrec)
            mean_accs.append(spectop1)
            means.append(specrec * 100)
        if tb_writer is not None:
            tb_writer.add_scalar("test/avg_spec_recall", mean(all_spec) * 100, epoch)
            tb_writer.add_scalar("test/avg_spec_top1_recall", mean(all_sp1) * 100, epoch)                
            
    prog.close()
    return means, all_accs, mean_accs
                                                       
def test_single_speconly_batch(test_loader, tb_writer, device, net, epoch):
    with tqdm(total=len(test_loader), unit="batch") as prog:
        means = []
        all_accs = []
        mean_accs = []
        for i, (specs_label, _, _, loaded_imgs) in enumerate(test_loader):
            batch = loaded_imgs.to(device)
            specs_lab = specs_label.to(device)                                     
            outputs = net(batch.float()) 
            spec_accs = utils.topk_acc(outputs, specs_label, topk=(30,1), device=device) # magic no from CELF2020
            prog.set_description("top 30: {acc0}  top1: {acc1}".format(acc0=spec_accs[0], acc1=spec_accs[1]))

            prog.update(1)          
            if tb_writer is not None:
                tb_writer.add_scalar("test/30_spec_accuracy", spec_accs[0], epoch)
                tb_writer.add_scalar("test/1_spec_accuracy", spec_accs[1], epoch)  

            all_accs.append(spec_accs)
            mean_accs.append(spec_accs)
            means.append(spec_accs.mean())
    prog.close()
    return means, all_accs, mean_accs



def train_batch(dataset, train_loader, device, optimizer, net, spec_loss, gen_loss, fam_loss, tb_writer, step, model, nepoch, epoch, loss):
    tot_loss_meter = []
    spec_loss_meter = []
    gen_loss_meter = []
    fam_loss_meter = []  
    with tqdm(total=len(train_loader), unit="batch") as prog:
        for i, ret in enumerate(train_loader):     
            specophs = nepoch
            genpoch = nepoch * 2
            fampoch = nepoch 
            # 



            # mixed data model MLP of environmental rasters + cnn of satellite imagery data
            if dataset == 'satellite_rasters_point':
                (specs_lab, gens_lab, fams_lab, batch, rasters) = ret
                if loss == 'all':
                        tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example_rasters(specs_lab, gens_lab, fams_lab, batch, rasters, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'all')
                elif loss == 'cumulative':
                    if epoch < fampoch:
                        # family only
                        tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example_rasters(specs_lab, gens_lab, fams_lab, batch, rasters, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'family')
                    elif epoch >= fampoch and epoch < genpoch:
                        # family and genus
                        tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example_rasters(specs_lab, gens_lab, fams_lab, batch, rasters, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'fam_gen') 
                    else:
                        # all 3 / spec only
                        tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example_rasters(specs_lab, gens_lab, fams_lab, batch, rasters, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'all')                    
                elif loss == 'sequential':
                    
                    if epoch < fampoch:
                        # family only
                        tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example_rasters(specs_lab, gens_lab, fams_lab, batch, rasters, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'family')
                    elif epoch >= fampoch and epoch < genpoch:
                        #  genus
                        tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example_rasters(specs_lab, gens_lab, fams_lab, batch, rasters, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'genus') 
                    else:
                        #  spec only
                        tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example_rasters(specs_lab, gens_lab, fams_lab, batch, rasters, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'species')
                elif loss == 'just_fam':
                    tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example_rasters(specs_lab, gens_lab, fams_lab, batch, rasters, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'family')
                elif loss == 'fam_gen':
                    tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example_rasters(specs_lab, gens_lab, fams_lab, batch, rasters, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'fam_gen')
                    # cnn model that goes straight from cnn to species outpute layer
                elif loss == 'just_spec':
                    tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example_rasters(specs_lab, gens_lab, fams_lab, batch, rasters, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'species')
                else: # loss is none, random forest baseline
                    raise NotImplemented
            elif model == 'SpecOnly':
                (specs_lab, gens_lab, fams_lab, batch) = ret
                tot_loss, loss_spec = forward_one_example_speconly(specs_lab, batch, optimizer, net, spec_loss, device)
                loss_fam, loss_gen = None, None
            elif model == 'MLP_Family':
                (specs_lab, gens_lab, fams_lab, batch) = ret
                tot_loss, loss_fam = forward_one_example_speconly(fams_lab, batch, optimizer, net, fam_loss, device)
                loss_spec, loss_gen = None, None
            elif model == 'MLP_Family_Genus':
                (specs_lab, gens_lab, fams_lab, batch) = ret
                tot_loss, loss_gen, loss_fam = forward_one_example_famgen(fams_lab, gens_lab, batch, optimizer, net, fam_loss, gen_loss, device)
                loss_spec =  None
            #if dataset != 'satellite_rasters_point:
            else:
                (specs_lab, gens_lab, fams_lab, batch) = ret
                if loss == 'all':
                    tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'all')
                elif loss == 'cumulative':
                    if epoch < fampoch:
                        # family only
                        tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'family')
                    elif epoch >= fampoch and epoch < genpoch:
                        # family and genus
                        tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'fam_gen')
                    else:
                        # all 3 
                        tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'all')
                
                elif loss == 'sequential':
                    if epoch < fampoch:
                        # family only
                        tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'family')
                    elif epoch >= fampoch and epoch < genpoch:
                        # family and genus
                        tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'genus')
                    else:
                        # all 3 
                        tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'species')
                
                elif loss == 'just_fam':
                    tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'family')
                elif loss == 'fam_gen':
                    tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'fam_gen')
                elif loss == 'just_spec':
                    tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example_speconly(specs_lab, batch, optimizer, net, spec_loss, device)
                else: # loss is none or one of the new options! None of the cumulative nonsense, just run loss normally
                    tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'all')
                
            if tb_writer is not None:
                if model == 'SpecOnly':
                    tb_writer.add_scalar("train/tot_loss", tot_loss, step)
                    tb_writer.add_scalar("train/spec_loss", loss_spec.item(), step)
                    tot_loss_meter.append(tot_loss.item())                
                    spec_loss_meter.append(loss_spec.item())
                elif model == 'MLP_Family':
                    tb_writer.add_scalar("train/tot_loss", tot_loss, step)
                    tb_writer.add_scalar("train/fam_loss", loss_fam.item(), step)
                    tot_loss_meter.append(tot_loss.item())                
                    fam_loss_meter.append(loss_fam.item())
                elif model == 'MLP_Family_Genus':
                    tb_writer.add_scalar("train/tot_loss", tot_loss, step)
                    tb_writer.add_scalar("train/fam_loss", loss_fam.item(), step)
                    tb_writer.add_scalar("train/gen_loss", loss_gen.item(), step)
                    tot_loss_meter.append(tot_loss.item())                
                    gen_loss_meter.append(loss_gen.item())
                    fam_loss_meter.append(loss_fam.item()) 
                else:
                    tb_writer.add_scalar("train/tot_loss", tot_loss, step)
                    tb_writer.add_scalar("train/spec_loss", loss_spec.item(), step)
                    tb_writer.add_scalar("train/fam_loss", loss_fam.item(), step)
                    tb_writer.add_scalar("train/gen_loss", loss_gen.item(), step)
                    tot_loss_meter.append(tot_loss.item())                
                    spec_loss_meter.append(loss_spec.item())
                    gen_loss_meter.append(loss_gen.item())
                    fam_loss_meter.append(loss_fam.item()) 
            prog.set_description("loss: {tot_loss}".format(tot_loss=tot_loss))
            prog.update(1)                
            step += 1
            
    return tot_loss_meter, spec_loss_meter, gen_loss_meter, fam_loss_meter, step    


def forward_one_example_rasters(specs_lab, gens_lab, fams_lab, batch, rasters, optimizer, net, spec_loss, gen_loss, fam_loss, device, calculated):
    batch = batch.to(device)
    rasters = rasters.to(device)
    specs_lab = specs_lab.to(device)                                     
    gens_lab = gens_lab.to(device)
    fams_lab = fams_lab.to(device)
    optimizer.zero_grad()
    (specs, gens, fams) = net(batch.float(), rasters.float()) 
    loss_spec = spec_loss(specs, specs_lab) 
    loss_gen = gen_loss(gens, gens_lab) 
    loss_fam = fam_loss(fams, fams_lab)       
    if calculated == 'species':
        total_loss = loss_spec
    elif calculated == 'family':
        total_loss = loss_fam
    elif calculated == 'genus':
        total_loss = loss_gen
    elif calculated == 'fam_gen':
        total_loss = loss_gen + loss_fam
    else:
        total_loss = loss_spec + loss_gen + loss_fam
    total_loss.backward()
    optimizer.step()
    return total_loss, loss_spec, loss_gen, loss_fam

                                                       
def forward_one_example_speconly(specs_lab, batch, optimizer, net, spec_loss, device):
    batch = batch.to(device)
    specs_lab = specs_lab.to(device)                                     
    optimizer.zero_grad()
    specs = net(batch.float()) 
    loss_spec = spec_loss(specs, specs_lab) 
    total_loss = loss_spec
    total_loss.backward()
    optimizer.step()
    return total_loss, loss_spec

def forward_one_example_famgen(fams_lab, gens_lab, batch, optimizer, net, fams_loss, gens_loss, device):
    batch = batch.to(device)
    fams_lab = fams_lab.to(device)
    gens_lab = gens_lab.to(device)
    optimizer.zero_grad()
    fams, gens = net(batch.float()) 
    loss_fam = fams_loss(fams, fams_lab) 
    loss_gen = gens_loss(gens, gens_lab) 
    total_loss = loss_fam + loss_gen
    total_loss.backward()
    optimizer.step()
    return total_loss, loss_gen, loss_fam

def forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, calculated):

    batch = batch.to(device)
    specs_lab = specs_lab.to(device)
    gens_lab = gens_lab.to(device)
    fams_lab = fams_lab.to(device)
    optimizer.zero_grad()
    (specs, gens, fams) = net(batch.float()) 
    loss_spec = spec_loss(specs, specs_lab) 
    loss_gen = gen_loss(gens, gens_lab) 
    loss_fam = fam_loss(fams, fams_lab)
    if calculated == 'species':
        total_loss = loss_spec
    elif calculated == 'family':
        total_loss = loss_fam
    elif calculated == 'genus':
        total_loss = loss_gen
    elif calculated == 'fam_gen':
        total_loss = loss_gen + loss_fam
    else:
        total_loss = loss_spec + loss_gen + loss_fam
    total_loss.backward()
    optimizer.step()
    return total_loss, loss_spec, loss_gen, loss_fam


def train_model(ARGS, params):

    print("torch version {}".format(torch.__version__))
#     print(params.params.device, ARGS.device, " hello3")
    print("number of devices visible: {dev}".format(dev=torch.cuda.device_count()))
    device = torch.device("cuda:{dev}".format(dev=ARGS.device) if ARGS.device  >= 0 else "cpu")
    print('using device: {device}'.format(device=device))
    if ARGS.device >= 0:
        print("current device: {dev} current device name: {name}".format(dev=torch.cuda.current_device(), name=torch.cuda.get_device_name(torch.cuda.current_device())))
    print("current host: {host}".format(host=socket.gethostname()))
    batch_size=params.params.batch_size
    n_epochs=ARGS.epoch
    # load observation data
    print("loading data")
    datick = time.time()
    train_dataset = setup_dataset(params.params.observation, ARGS.base_dir, params.params.organism, params.params.region, params.params.normalize, params.params.no_altitude, params.params.dataset, params.params.threshold)
    if not ARGS.toy_dataset:
        tb_writer = SummaryWriter(comment="_lr-{}_mod-{}_reg-{}_obs-{}_dat-{}org-{}_loss-{}_norm-{}_exp_id-{}".format(params.params.lr, params.params.model, params.params.region, params.params.observation, params.params.dataset, params.params.organism, params.params.loss, params.params.normalize, params.params.exp_id))

    else:
        tb_writer = None
#         train_dataset.obs = train_dataset.obs[:params.batch_size*2]
    val_split = .9
    print("setting up network")
    # global so can access in collate_fn easily
    global num_specs 
    num_specs = train_dataset.num_specs
    global num_fams
    num_fams = train_dataset.num_fams
    global num_gens
    num_gens = train_dataset.num_gens    
    start_epoch = None
    step = None 
    net = setup_model(params.params.model, train_dataset)
    net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=params.params.lr)
    
    if ARGS.from_scratch or not ARGS.load_from_config:
        start_epoch = 0
        step = 0         
        train_samp, test_samp, idxs = better_split_train_test(train_dataset, val_split)        
    else:
        net_load = params.get_recent_model(device=device)
        net.load_state_dict(net_load['model_state_dict'])
        net.to(device)
        optimizer = optim.Adam(net.parameters(), lr=params.params.lr)
        optimizer.load_state_dict(net_load['optimizer_state_dict'])
        start_epoch = net_load['epoch']
        step = net_load['step']
        print("loading model from epoch {}".format(start_epoch))
        train_idx, test_idx = train_dataset.train, train_dataset.test
        train_samp = SubsetRandomSampler(train_idx)
        test_samp = SubsetRandomSampler(test_idx) 
        idxs = {'train' : train_idx, 'test' : test_idx}
        desi = params.get_most_recent_des()
        train_dataset.inv_spec = desi['inv_spec']
        train_dataset.spec_dict = desi['spec_dict']
        train_dataset.gen_dict = desi['gen_dict']
        train_dataset.fam_dict = desi['fam_dict']        

        
        
    spec_loss, gen_loss, fam_loss = setup_loss(params.params.observation, train_dataset, params.params.loss, params.params.unweighted, device, params.params.loss_type) 

    if ARGS.toy_dataset:

        test_dataset = copy.deepcopy(train_dataset)
        train_dataset.obs = train_dataset.obs[:1000]
        test_dataset.obs = test_dataset.obs[1000:2000]
        train_loader = setup_dataloader(train_dataset, params.params.dataset, batch_size, ARGS.processes,  SubsetRandomSampler(np.arange(1000)), ARGS.model)
        test_loader = setup_dataloader(test_dataset, params.params.dataset, batch_size, ARGS.processes, SubsetRandomSampler(np.arange(1000)), ARGS.model)
        
    else:
        train_loader = setup_dataloader(train_dataset, params.params.dataset, batch_size, ARGS.processes, train_samp, ARGS.model)
        test_loader = setup_dataloader(train_dataset, params.params.dataset, batch_size, ARGS.processes, test_samp, ARGS.model)
 

    
    datock = time.time()
    dadiff = datock - datick
    print("loading data took {dadiff} seconds".format(dadiff=dadiff))
    print("number of channels are {}".format(train_dataset.channels))
    num_batches = math.ceil(len(train_dataset) / batch_size)
    print("batch size is {batch_size} and size of dataset is {lens} and num batches is {num_batches}\n".format(batch_size=batch_size, lens=len(train_dataset), num_batches=len(train_loader)))
    print("starting training") 
    all_time_loss = []
    all_time_sp_loss = []
    all_time_gen_loss = []
    all_time_fam_loss = []  
    
    if params.params.loss == 'sequential' or params.params.loss == 'cumulative':
        tot_epoch = n_epochs
        n_epochs = n_epochs*3
        print("total number of epochs: {} number of epochs of each loss type: {}", n_epochs, tot_epoch)
    else:
        tot_epoch = n_epochs
    epoch = start_epoch
    while epoch < n_epochs:
        print("starting training for epoch {}".format(epoch))
        tick = time.time()
        net.train()
        print("before batch")
        
        tot_loss_meter, spec_loss_meter, gen_loss_meter, fam_loss_meter, step = train_batch(params.params.observation, train_loader, device, optimizer, net, spec_loss, gen_loss, fam_loss, tb_writer, step, params.params.model, tot_epoch, epoch, params.params.loss)
        print('after batch')
        #TODO change back!!!
        if not ARGS.toy_dataset:

            if params.params.model == 'SpecOnly':
                all_time_loss.append(np.stack(tot_loss_meter))
                all_time_sp_loss.append(np.stack(spec_loss_meter))
                all_time_gen_loss = []
                all_time_fam_loss =[]
            elif params.params.model == 'MLP_Family':
                all_time_loss.append(np.stack(tot_loss_meter))
                all_time_sp_loss = []
                all_time_gen_loss = []
                all_time_fam_loss.append(np.stack(fam_loss_meter))
            elif params.params.model == 'MLP_Family_Genus':
                all_time_loss.append(np.stack(tot_loss_meter))
                all_time_sp_loss = []
                all_time_gen_loss.append(np.stack(gen_loss_meter))
                all_time_fam_loss.append(np.stack(fam_loss_meter))
            else:
                all_time_loss.append(np.stack(tot_loss_meter))
                all_time_sp_loss.append(np.stack(spec_loss_meter))
                all_time_gen_loss.append(np.stack(gen_loss_meter))
                all_time_fam_loss.append(np.stack(fam_loss_meter))
            nets_path=params.build_abs_nets_path(epoch)
            print('nets path ', nets_path)
            torch.save({
                        'epoch': epoch,
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'step' : step
                        }, nets_path)        

        # test
        net.eval()
        all_accs = []
        print("testing model")
        with torch.no_grad():
            means, all_accs, mean_accs = test_batch(test_loader, tb_writer, device, net, params.params.observation, epoch, params.params.loss, params.params.model, params.params.dataset)

#TODO uncomment below line!!!
        if not ARGS.toy_dataset:
            desiderata = {
                'all_loss': all_time_loss,
                'spec_loss': all_time_sp_loss,
                'gen_loss': all_time_gen_loss,
                'fam_loss': all_time_fam_loss,
                'means': means,
                'all_accs': all_accs,
                'mean_accs': mean_accs,
                'splits' : idxs,
                'batch_size': batch_size,
                'inv_spec' : train_dataset.inv_spec, 
                'spec_dict' : train_dataset.spec_dict, 
                'gen_dict' : train_dataset.gen_dict, 
                'fam_dict': train_dataset.fam_dict
            }
            desiderata_path = params.build_abs_desider_path(epoch)
            with open(desiderata_path, 'wb') as f:
                pickle.dump(desiderata, f)
#         insert inference again here???
            inference.eval_model(params.get_cfg_name()+'.json', ARGS.base_dir, ARGS.toy_dataset, epoch=epoch)
        tock = time.time()
        diff = ( tock-tick)/60
        print ("epoch {} took {} minutes".format(epoch, diff))
        epoch += 1
    if not ARGS.toy_dataset:
        tb_writer.close()

if __name__ == "__main__":
    args = ['load_from_config','lr', 'epoch', 'device', 'toy_dataset', 'loss', 'processes', 'exp_id', 'base_dir', 'region', 'organism', 'seed', 'observation', 'batch_size', 'model', 'normalize', 'unweighted', 'no_alt', 'from_scratch', 'dataset', 'threshold', 'loss_type']
    ARGS = config.parse_known_args(args)       
    config.setup_main_dirs(ARGS.base_dir)
    print(ARGS)
    print('epoch', ARGS.epoch)
    print('load from config ', ARGS.load_from_config)
    if ARGS.load_from_config is not None:
        params = config.Run_Params(ARGS.base_dir, ARGS)
    else:
        params = config.Run_Params(ARGS.base_dir, ARGS)

    print(type(params), type(ARGS))
    params.setup_run_dirs(ARGS.base_dir)
    train_model(ARGS, params)
