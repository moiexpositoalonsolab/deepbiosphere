import pickle
import pandas as pd
import argparse
import time
#from IPython.core.debugger import set_trace
import deepdish as dd
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
from deepbiosphere.scripts import GEOCLEF_Dataset as Dataset
from deepbiosphere.scripts import GEOCLEF_Utils as utils
from deepbiosphere.scripts import GEOCLEF_Config as config





def split_train_test(full_dat, split_amt):
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


def setup_train_dataset(observation, base_dir, organism, region, toy_dataset, normalize, model):
    '''grab and setup train dataset'''

    if observation == 'single':
        if region == 'both':
            return Dataset.GEOCELF_Dataset_Full(base_dir, organism)
        else:
            return Dataset.Single_Toy_Dataset(base_dir, organism, region)  if toy_dataset else Dataset.GEOCELF_Dataset(base_dir, organism, region)
    elif observation == 'joint':
        if region == 'both':
            return Dataset.GEOCELF_Dataset_Joint_Full(base_dir, organism)
        else:
            if model == 'MixNet' or model == 'MixFullNet':
                return Dataset.GEOCELF_Dataset_Joint_BioClim(base_dir, organism, region, normalize=normalize) 
            else:
                return Dataset.Joint_Toy_Dataset(base_dir, organism, region) if toy_dataset else Dataset.GEOCELF_Dataset_Joint(base_dir, organism, region)
    else:
        exit(1), "should never reach this..."
        
def setup_model(model, num_specs, num_fams, num_gens, num_channels, num_rasters=None):
    # baselines
    if model == 'SVM':
        raise NotImplementedError
    elif model == 'RandomForest':
        raise NotImplementedError
    elif model == 'SVM':
        raise NotImplementedError
    # some flavor of convnet architecture
    elif model == 'OGNoFamNet':
        return cnn.OGNoFamNet(species=num_specs, families=num_fams, genuses=num_gens, num_channels=num_channels)        
    elif model == 'OGNet':
        return cnn.OGNet(species=num_specs, families=num_fams, genuses=num_gens, num_channels=num_channels)        
    elif model == 'SkipFCNet':
        return cnn.SkipFCNet(species=num_specs, families=num_fams, genuses=num_gens, num_channels=num_channels)        
    elif model == 'SkipNet':
        return cnn.SkipNet(species=num_specs, families=num_fams, genuses=num_gens, num_channels=num_channels)        
    elif model == 'MixNet':
        return cnn.MixNet(species=num_specs, families=num_fams, genuses=num_gens, num_channels=num_channels, env_rasters=num_rasters)
    elif model == 'MixFullNet':
        return cnn.MixFullNet(species=num_specs, families=num_fams, genuses=num_gens, num_channels=num_channels, env_rasters=num_rasters)
    elif model == 'SkipFullFamNet':
        return cnn.SkipFullFamNet(species=num_specs, families=num_fams, genuses=num_gens, num_channels=num_channels)
    else: 
        exit(1), "if you reach this, you got a real problem bucko"



def setup_GeoCLEF_dataloaders(train_dataset, base_dir, region, observation, batch_size, processes, optimizer, net, spec_loss, fam_loss, gen_loss):
    train_samp = SubsetRandomSampler(np.arange(len(train_dataset)))
    train_loader = setup_dataloader(train_dataset, observation, batch_size, processes, train_samp)
    #Call check_batch_size here!
    if ARGS.dynamic_batch:
        batch_size, train_loader = check_batch_size(observation, train_dataset, processes, train_loader, batch_size, optimizer, net, spec_loss, fam_loss, gen_loss, train_samp, device)
    test_dataset = Dataset.GEOCELF_Test_Dataset_Full(base_dir) if region == 'us_fr' else Dataset.GEOCELF_Test_Dataset(base_dir, region)
    test_loader = DataLoader(test_dataset, batch_size, shuffle=True, pin_memory=False, num_workers=processes)       
    return train_loader, test_loader, batch_size

def setup_dataloader(dataset, observation, batch_size, processes, sampler, model):
    if model == 'MixNet' or model == 'MixFullNet':
        collate_fn = joint_raster_collate_fn
    elif observation == 'joint':
        collate_fn = joint_collate_fn
    elif observation == 'single':
        collate_fn = single_collate_fn
    dataloader = DataLoader(dataset, batch_size, pin_memory=False, num_workers=processes, collate_fn=collate_fn, sampler=sampler)

    return dataloader


    
def setup_loss(observation, dataset, device):
    spec_freq = Dataset.freq_from_dict(dataset.spec_freqs)
    gen_freq = Dataset.freq_from_dict(dataset.gen_freqs)
    fam_freq = Dataset.freq_from_dict(dataset.fam_freqs)    
    spec_freq = 1.0 / torch.tensor(spec_freq, dtype=torch.float, device=device)
    gen_freq = 1.0 / torch.tensor(gen_freq, dtype=torch.float, device=device)
    fam_freq = 1.0 / torch.tensor(fam_freq, dtype=torch.float, device=device)
    if observation == 'joint':
        spec_loss = torch.nn.BCEWithLogitsLoss(spec_freq)
        gen_loss = torch.nn.BCEWithLogitsLoss(gen_freq)
        fam_loss = torch.nn.BCEWithLogitsLoss(fam_freq)    
        return spec_loss, gen_loss, fam_loss
    elif observation == 'single':
        spec_loss = torch.nn.CrossEntropyLoss(spec_freq)
        gen_loss = torch.nn.CrossEntropyLoss(gen_freq)
        fam_loss = torch.nn.CrossEntropyLoss(fam_freq)  
        return spec_loss, gen_loss, fam_loss
    
    
def clean_gpu(device):
    if device is not None:
        print("cleaning gpu")            
        torch.cuda.empty_cache()
        
def single_collate_fn(batch): 
    # batch is a list of tuples of (composite_label <np array [3]>, images <np array [6, 256, 256]>)   
    labs, img = zip(*batch) 
    lbs = [torch.tensor(l, dtype=torch.long) for l in labs]      
    img = [i.astype(np.uint8, copy=False) for i in img]
    imgs = [torch.from_numpy(i) for i in img]
    return torch.stack(lbs), torch.stack(imgs)  

def joint_collate_fn(batch):
    # batch is a list of tuples of (specs_label, gens_label, fams_label, images)  
    all_specs = []
    all_gens = []
    all_fams = []
    imgs = []
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

def test_batch(test_loader, tb_writer, device, net, observation, epoch, model):
    if model == 'MixNet' or model == 'MixFullNet':
        return test_joint_obs_rasters_batch(test_loader, tb_writer, device, net, epoch)
    elif observation == 'joint':
        return test_joint_obs_batch(test_loader, tb_writer, device, net, epoch)
    elif observation == 'single':
        return test_single_obs_batch(test_loader, tb_writer, device, net, epoch)

def test_single_obs_batch(test_loader, tb_writer, device, net, epoch):
     with tqdm(total=len(test_loader), unit="batch") as prog:
        all_accs = []
        all_spec = []
        all_gen = []
        all_fam = []
        for i, (labels, batch) in enumerate(test_loader):
            labels = labels.to(device)
            batch = batch.to(device)
            (outputs, genus, family) = net(batch.float())
            spec_accs = utils.topk_acc(outputs, labels[:,0], topk=(30,1), device=device) # magic no from CELF2020
            gens_accs = utils.topk_acc(genus, labels[:,1], topk=(30,1), device=device) # magic no from CELF2020
            fam_accs = utils.topk_acc(family, labels[:,2], topk=(30,1), device=device) # magic no from CELF2020
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
    
def test_joint_obs_rasters_batch(test_loader, tb_writer, device, net, epoch):
    with tqdm(total=len(test_loader), unit="batch") as prog:
        means = []
        all_accs = []
        mean_accs = []
        for i, (specs_label, gens_label, fams_label, imgs, env_rasters) in enumerate(test_loader):
            imgs = imgs.to(device)
            env_rasters = env_rasters.to(device)
            specs_lab = specs_label.to(device)                                     
            gens_label = gens_label.to(device)
            fams_label = fams_label.to(device)
            (outputs, gens, fams) = net(imgs.float(), env_rasters.float()) 
            specaccs, totspec_accs = utils.num_corr_matches(outputs, specs_lab) # magic no from CELF2020
            genaccs, totgen_accs = utils.num_corr_matches(gens, gens_label) # magic no from CELF2020                        
            famaccs, totfam_accs = utils.num_corr_matches(fams, fams_label) # magic no from CELF2020                        
            prog.set_description("mean accuracy across batch: {acc0}".format(acc0=specaccs.mean()))
            prog.update(1)          
            if tb_writer is not None:
                tb_writer.add_scalar("test/avg_spec_accuracy", specaccs.mean(), epoch)
                tb_writer.add_scalar("test/avg_gen_accuracy", genaccs.mean(), epoch)
                tb_writer.add_scalar("test/avg_fam_accuracy", famaccs.mean(), epoch)                        
            else:
                break
            all_accs.append(totspec_accs)
            mean_accs.append(specaccs)
            means.append(specaccs.mean()) 
    prog.close()
    return means, all_accs, mean_accs
def test_joint_obs_batch(test_loader, tb_writer, device, net, epoch):
    with tqdm(total=len(test_loader), unit="batch") as prog:
        means = []
        all_accs = []
        mean_accs = []
        for i, (specs_label, gens_label, fams_label, loaded_imgs) in enumerate(test_loader):
            batch = loaded_imgs.to(device)
            specs_lab = specs_label.to(device)                                     
            gens_label = gens_label.to(device)
            fams_label = fams_label.to(device)
            (outputs, gens, fams) = net(batch.float()) 
            specaccs, totspec_accs = utils.num_corr_matches(outputs, specs_lab) # magic no from CELF2020
            genaccs, totgen_accs = utils.num_corr_matches(gens, gens_label) # magic no from CELF2020                        
            famaccs, totfam_accs = utils.num_corr_matches(fams, fams_label) # magic no from CELF2020                        
            prog.set_description("mean accuracy across batch: {acc0}".format(acc0=specaccs.mean()))
            prog.update(1)          
            if tb_writer is not None:
                tb_writer.add_scalar("test/avg_spec_accuracy", specaccs.mean(), epoch)
                tb_writer.add_scalar("test/avg_gen_accuracy", genaccs.mean(), epoch)
                tb_writer.add_scalar("test/avg_fam_accuracy", famaccs.mean(), epoch)                        
            all_accs.append(totspec_accs)
            mean_accs.append(specaccs)
            means.append(specaccs.mean()) 
    prog.close()
    return means, all_accs, mean_accs

def write_batch_GeoCLEF(batch, device, net, writer, train_dataset):
    batch = batch.to(device)                                  
    (outputs, _, _) = net(batch.float()) 
    scores, idxs = torch.topk(outputs.cpu(), dim=1, k=150)
    top_scores = scores[:,:150]
    top_idxs = idxs[:,:150]
    for scores, ids, idd in zip(top_scores, top_idxs, id_):
        ids = [train_dataset.idx_2_id[i.item()] for i in ids]
        row = itertools.chain( [idd.item()], ids, scores.tolist())
        writer.writerow(row)

def test_GeoCLEF_batch(test_loader, base_dir, region, exp_id, epoch):
    with tqdm(total=len(test_loader), unit="obs") as prog:
        file = "{}output/{}_{}_e{}.csv".format(base_dir, country, exp_id, epoch)
        with open(file,'w') as f:
            writer = csv.writer(f, dialect='unix')
            top_class = ['top_{n}_class_id'.format(n=n) for n in np.arange(1, 151)] #GeoCLEF magic number
            top_score = ['top_{n}_class_score'.format(n=n) for n in np.arange(1, 151)]  
            header = ['observation_id'] + top_class + top_score
            writer.writerow(header)
            for i, (batch, id_) in enumerate(test_loader):
                write_batch_GeoCLEF(batch, device, net, writer, train_dataset)
                prog.update(1)
    prog.close()

def train_batch(observation, train_loader, device, optimizer, net, spec_loss, gen_loss, fam_loss, tb_writer, step, model, species_only, sequential, cumulative, nepoch, epoch):
    
    tot_loss_meter = []
    spec_loss_meter = []
    gen_loss_meter = []
    fam_loss_meter = []  
#     print(f"epoch {epoch} num epochs {nepoch}")
#     print(f"species only {species_only} sequential {sequential} cumulative {cumulative}")
    with tqdm(total=len(train_loader), unit="batch") as prog:
        
        for i, ret in enumerate(train_loader):     
            if model == 'MixNet' or model == 'MixFullNet':
                (specs_lab, gens_lab, fams_lab, batch, rasters) = ret
                if species_only:
#                     print("species only loss")
                    tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example_rasters(specs_lab, gens_lab, fams_lab, batch, rasters, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'species')
                elif sequential or cumulative:
                    specophs = nepoch
                    genpoch = nepoch * 2
                    fampoch = nepoch 
#                     print(f"\n epoch {epoch} genpoch {genpoch} fampoch {fampoch}")    
#                     print(f" epoch < fampoch {epoch < fampoch} epoch > fampoch and epoch < genpoch {epoch > fampoch and epoch < genpoch}  ")
                    if epoch < fampoch:
                        # family only
#                         print("family only loss")
                        tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example_rasters(specs_lab, gens_lab, fams_lab, batch, rasters, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'family')
                    elif epoch >= fampoch and epoch < genpoch:
                        # family and genus
#                         print("family and / or genus only loss")
                        tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example_rasters(specs_lab, gens_lab, fams_lab, batch, rasters, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'fam_gen') if cumulative else forward_one_example_rasters(specs_lab, gens_lab, fams_lab, batch, rasters, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'genus')
                    else:
                        # all 3 / spec only
#                         print("all or species loss ")
                        tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example_rasters(specs_lab, gens_lab, fams_lab, batch, rasters, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'all') if cumulative else forward_one_example_rasters(specs_lab, gens_lab, fams_lab, batch, rasters, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'species')
                else:
                    tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example_rasters(specs_lab, gens_lab, fams_lab, batch, rasters, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'all')               

            elif observation == 'joint':

                (specs_lab, gens_lab, fams_lab, batch) = ret
                
                if species_only:
                    
                    tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'species')
                elif sequential or cumulative:
                    genpoch = nepoch * 2
                    fampoch = nepoch 
#                     print(f"\n epoch {epoch} genpoch {genpoch} fampoch {fampoch}")    
#                     print(f" epoch < fampoch {epoch < fampoch} epoch > fampoch and epoch < genpoch {epoch > fampoch and epoch < genpoch}  ")                  
                    if epoch < fampoch:
                        # family only
#                         print("family only loss")                        
                        tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'family')
                    elif epoch >= fampoch and epoch < genpoch:
                        # family and genus
#                         print("family and / or genus only loss")                        
                        tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'fam_gen') if cumulative else forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'genus')
                    else:
                        # all 3 / spec only
#                         print("all or species loss ")                        
                        tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'all') if cumulative else forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'species')
                else:
                    tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'all')               

            elif observation == 'single':
                (labels, batch) = ret
                specs_lab = labels[:,0]
                gens_lab = labels[:,1]
                fams_lab = labels[:,2]
                tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'species') if species_only else forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device, 'all')
#             tot_loss, loss_spec, loss_gen, loss_fam = forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device)
            if tb_writer is not None:
                tb_writer.add_scalar("train/tot_loss", tot_loss, step)
                tb_writer.add_scalar("train/spec_loss", loss_spec.item(), step)
                tb_writer.add_scalar("train/fam_loss", loss_fam.item(), step)
                tb_writer.add_scalar("train/gen_loss", loss_gen.item(), step)
            else:
                break
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
    #print(" total loss {}, total spec loss {}, genus loss {}, family loss {}".format(total_loss.item(), loss_spec.item(), loss_gen.item(), loss_fam.item()))
    total_loss.backward()
    optimizer.step()
    return total_loss, loss_spec, loss_gen, loss_fam

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
#     print(f"calculated is {calculated}")
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
#     print(f" total loss {total_loss.item()}, total spec loss {loss_spec.item()}, genus loss {loss_gen.item()}, family loss {loss_fam.item()}")
    total_loss.backward()
    optimizer.step()
    return total_loss, loss_spec, loss_gen, loss_fam


def check_batch_size(observation, dataset, processes, loader, batch_size, optimizer, net, spec_loss, fam_loss, gen_loss, sampler, device):
    if batch_size < 1:
        exit(1), "non-memory error present!"
    for ret in loader:
        if observation == 'single':
            labels, batch = ret
            specs_lab = labels[:,0]
            gens_lab = labels[:,1]
            fams_lab = labels[:,2]
        elif observation == 'joint':
            specs_lab, gens_lab, fams_lab, batch = ret
        try:
            forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device)
        except RuntimeError:
            batch_size -= 1
            print("decreasing batch size to {}".format(batch_size))
            loader = setup_dataloader(dataset, observation, batch_size, processes, sampler)
            check_batch_size(observation, dataset, processes, loader, batch_size, optimizer, net, spec_loss, fam_loss, gen_loss, sampler, device)
        break
    return batch_size, loader
    
def train_model(ARGS, params):
    #TODO: add checking if config exists and starting from that version
    print("torch version {}".format(torch.__version__))
#     print(params.params.device, ARGS.device, " hello3")
    print("number of devices visible: {dev}".format(dev=torch.cuda.device_count()))
    device = torch.device("cuda:{dev}".format(dev=params.params.device) if params.params.device  > 0 else "cpu")
    print('using device: {device}'.format(device=device))
    if params.params.device > 0:
        print("current device: {dev} current device name: {name}".format(dev=torch.cuda.current_device(), name=torch.cuda.get_device_name(torch.cuda.current_device())))
    print("current host: {host}".format(host=socket.gethostname()))
    batch_size=ARGS.batch_size
    n_epochs=ARGS.epoch
    # load observation data
    print("species only: {} cumulative: {} sequential: {}".format(ARGS.species_only, ARGS.cumulative, ARGS.sequential))
    print("loading data")
    datick = time.time()
    train_dataset = setup_train_dataset(params.params.observation, ARGS.base_dir, params.params.organism, params.params.region, ARGS.toy_dataset, ARGS.normalize, ARGS.model)
    if not ARGS.toy_dataset:
        tb_writer = SummaryWriter(comment="_lr-{}_mod-{}_reg-{}_obs-{}_org-{}_exp_id-{}".format(params.params.lr, params.params.model, params.params.region, params.params.observation, params.params.organism, params.params.exp_id))

    else:
        tb_writer = None
    val_split = .9
    print("setting up network")
    # global so can access in collate_fn easily
    global num_specs 
    num_specs = train_dataset.num_specs
    global num_fams
    num_fams = train_dataset.num_fams
    global num_gens
    num_gens = train_dataset.num_gens    
    num_channels = train_dataset.channels
    num_rasters = train_dataset.num_rasters if ARGS.model == 'MixNet' or ARGS.model == 'MixFullNet' else None
    start_epoch = None
    net_path = params.get_recent_model(ARGS.base_dir)
    
    if net_path is None or ARGS.from_scratch:
        net = setup_model(params.params.model, num_specs, num_fams, num_gens, num_channels, num_rasters)
        optimizer = optim.Adam(net.parameters(), lr=params.params.lr)
        net.to(device)
        start_epoch = 0
        step = 0 
    else:
        model = torch.load(net_path, map_location=device)
        net = setup_model(params.params.model, num_specs, num_fams, num_gens, num_channels, num_rasters)
        optimizer = optim.Adam(net.parameters(), lr=params.params.lr)
        net.load_state_dict(model['model_state_dict'])
        optimizer.load_state_dict(model['optimizer_state_dict'])
        start_epoch = model['epoch']
        start_step = model['step']
        net.to(device)
#         optimizer.to(device)
        print("loading model from epoch {}".format(start_epoch))
    
    spec_loss, gen_loss, fam_loss = setup_loss(params.params.observation, train_dataset, device) 
#     set_trace()       
    if ARGS.GeoCLEF_validate:
        train_loader, test_loader, batch_size = setup_GeoCLEF_dataloaders(train_dataset, ARGS.base_dir, params.params.region, params.params.observation, batch_size, ARGS.processes, optimizer, net, spec_loss, fam_loss, gen_loss)

    else: 
        train_samp, test_samp, idxs = split_train_test(train_dataset, val_split) 
        train_loader = setup_dataloader(train_dataset, params.params.observation, batch_size, ARGS.processes, train_samp, ARGS.model)
        if ARGS.dynamic_batch:
            batch_size, train_loader = check_batch_size(params.params.observation, train_dataset, ARGS.processes, train_loader, batch_size, optimizer, net, spec_loss, fam_loss, gen_loss, train_samp, device)
        test_loader = setup_dataloader(train_dataset, params.params.observation, batch_size, ARGS.processes, test_samp, ARGS.model)
    datock = time.time()
    dadiff = datock - datick
    print("loading data took {dadiff} seconds".format(dadiff=dadiff))
#     import pdb; pdb.set_trace()
#     print("model size is ", num_rasters)
#     for tens in next(iter(train_loader)):
#         print(tens.shape)
#     print("model size is ", num_rasters, "data size is ", next(iter(train_loader)) )
#     exit(1)
    num_batches = math.ceil(len(train_dataset) / batch_size)
    print("batch size is {batch_size} and size of dataset is {lens} and num batches is {num_batches}\n".format(batch_size=batch_size, lens=len(train_dataset), num_batches=len(train_loader)))
    print("starting training") 
    all_time_loss = []
    all_time_sp_loss = []
    all_time_gen_loss = []
    all_time_fam_loss = []  
#     step = 0
    if ARGS.cumulative or ARGS.sequential:
        tot_epoch = n_epochs
        n_epochs = n_epochs*3
    else:
        tot_epoch = n_epochs
    epoch = start_epoch
    while epoch < n_epochs:
        print("starting training for epoch {}".format(epoch))
        clean_gpu(device)
        #if params.params.device is not None:
            #torch.cuda.synchronize()
        tick = time.time()
        net.train()
        if ARGS.dynamic_batch:
            try:
                tot_loss_meter, spec_loss_meter, gen_loss_meter, fam_loss_meter, step = train_batch(params.params.observation, train_loader, device, optimizer, net, spec_loss, gen_loss, fam_loss, tb_writer, step, params.params.model, ARGS.species_only, ARGS.sequential, ARGS.cumulative, tot_epoch, epoch)
            except RuntimeError: # CUDA: Out of memory is a generic RTE
                batch_size -= 1
                if batch_size < 0:
                    exit(1), "non-memory error present!"
                print("decreasing batch size to {} at epoch {}".format(batch_size, epoch))
                if ARGS.GeoCLEF_validate:
                    train_loader, test_loader, batch_size = setup_GeoCLEF_dataloaders(train_dataset, ARGS.base_dir, params.params.region, params.params.observation, batch_size, ARGS.processes, optimizer, net, spec_loss, fam_loss, gen_loss)
                else:
                    train_loader = setup_dataloader(train_dataset, params.params.observation, batch_size, ARGS.processes, train_samp)       

                    test_loader = setup_dataloader(train_dataset, params.params.observation, batch_size, ARGS.processes, test_samp)
                    print(check_mem())
                    continue
        else:
            tot_loss_meter, spec_loss_meter, gen_loss_meter, fam_loss_meter, step = train_batch(params.params.observation, train_loader, device, optimizer, net, spec_loss, gen_loss, fam_loss, tb_writer, step, params.params.model, ARGS.species_only, ARGS.sequential, ARGS.cumulative, tot_epoch, epoch)
        if not ARGS.toy_dataset:
            all_time_loss.append(np.stack(tot_loss_meter))
            all_time_sp_loss.append(np.stack(spec_loss_meter))
            all_time_gen_loss.append(np.stack(gen_loss_meter))
            all_time_fam_loss.append(np.stack(fam_loss_meter))        
        
        # save model 
        nets_path=params.build_abs_nets_path(ARGS.base_dir, epoch)
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step' : step
                    }, nets_path)        
        
        clean_gpu(device)
        # test
        net.eval()
        all_accs = []
        print("testing model")
        with torch.no_grad():
            if ARGS.GeoCLEF_validate:
                test_GeoCLEF_batch(test_loader, ARGS.base_dir, params.params.region, params.params.exp_id, epoch)
            else:
                means, all_accs, mean_accs = test_batch(test_loader, tb_writer, device, net, params.params.observation, epoch, params.params.model)       
        clean_gpu(device)        
        desiderata = {
            'all_loss': all_time_loss,
            'spec_loss': all_time_sp_loss,
            'gen_loss': all_time_gen_loss,
            'fam_loss': all_time_fam_loss,
            'means': means,
            'all_accs': all_accs,
            'mean_accs': mean_accs,
            'splits' : idxs
        }
        desiderata_path = params.build_abs_desider_path(ARGS.base_dir, epoch)
        #dd.io.save(desiderata_path, desiderata, compression=True)
        with open(desiderata_path, 'wb') as f:
            pickle.dump(desiderata, f, protocol=pickle.HIGHEST_PROTOCOL)
        tock = time.time()
        diff = ( tock-tick)/60
        print ("one epoch took {} minutes".format(diff))
        epoch += 1
#         print("epoch is {}".format(epoch))
    if not ARGS.toy_dataset:
        tb_writer.close()

if __name__ == "__main__":
    args = ['lr', 'epoch', 'device', 'toy_dataset', 'processes', 'exp_id', 'base_dir', 'region', 'organism', 'seed', 'GeoCLEF_validate', 'observation', 'batch_size', 'model', 'from_scratch', 'dynamic_batch', 'normalize', 'species_only', 'sequential', 'cumulative']
#     print("main ", args)
    ARGS = config.parse_known_args(args)
    # TODO: fix!
    if ARGS.species_only:
        ARGS.exp_id = "{}_{}".format(ARGS.exp_id, 'spec_only')
    elif ARGS.cumulative:
        ARGS.exp_id = "{}_{}".format(ARGS.exp_id, 'cumulative')
    elif ARGS.sequential:
        ARGS.exp_id = "{}_{}".format(ARGS.exp_id, 'sequential')        
    config.setup_main_dirs(ARGS.base_dir)
    params = config.Run_Params(ARGS)
    params.setup_run_dirs(ARGS.base_dir)
    train_model(ARGS, params)
