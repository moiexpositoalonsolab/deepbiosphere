
from random import randrange
import statistics
import pandas as pd
import itertools
import os
import argparse
import time
import numpy as np
import socket
import torch
import torch.nn as nn
import gc
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import csv
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random
import math
from tqdm import tqdm
from deepbiosphere.scripts import GEOCELF_CNN as cnn
from deepbiosphere.scripts import GEOCELF_Dataset as Dataset
from deepbiosphere.scripts import paths
        
    
def topk_acc(output, target):
    """Computes the precision@k for the specified values of k"""
    tot_acc = []
    acc_acc = []
    for obs, trg in zip(output, target):
        out_vals, out_idxs = torch.topk(obs, int(trg.sum().item()))
        targ_vals, targ_idxs = torch.topk(trg, int(trg.sum().item()))
        eq = len(list(set(out_idxs.tolist()) & set(targ_idxs.tolist())))
        acc = eq / trg.sum() * 100
        tot_acc.append((eq, len(targ_idxs)))
        acc_acc.append(acc.item())
    
    return np.stack(acc_acc), np.stack(tot_acc)


def check_mem():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.device)
        except: pass




def split_train_test(full_dat, split_amt):
    #grab split_amt% of labeled data for holdout testing
    idxs = np.random.permutation(len(full_dat))
    split = int(len(idxs)*split_amt)
    training_idx, test_idx = idxs[:split], idxs[split:]
    train_sampler = SubsetRandomSampler(training_idx)
    valid_sampler = SubsetRandomSampler(test_idx)
    return train_sampler, valid_sampler

def main():
    print("number of devices visible: {dev}".format(dev=torch.cuda.device_count()))
    device = torch.device("cuda:{dev}".format(dev=ARGS.device) if ARGS.device is not None else "cpu")
    print('using device: {device}'.format(device=device))
    if ARGS.device is not None:
        print("current device: {dev} current device name: {name}".format(dev=torch.cuda.current_device(), name=torch.cuda.get_device_name(torch.cuda.current_device())))
    print("current host: {host}".format(host=socket.gethostname()))

    # load observation data
    print("loading data")
    datick = time.time()
    if ARGS.country == 'both':
        train_dataset = Dataset.GEOCELF_Dataset_Joint_Full(ARGS.base_dir)
    else:
        #TODO: gross code
        split = 'plant' if ARGS.plants else 'train'
        train_dataset = Dataset.GEOCELF_Dataset_Joint(ARGS.base_dir, ARGS.country, split)
    val_split = .9
    tb_writer = SummaryWriter(comment="exp_id: {}".format(ARGS.exp_id))
    # set up net
    datock = time.time()
    dadiff = datock - datick
    print("loading data took {dadiff} seconds".format(dadiff=dadiff))
    print("setting up network")
    num_channels = train_dataset.channels# num_channels should be idx 1 in the order torch expects
    num_specs = train_dataset.num_specs
    num_fams = train_dataset.num_fams
    num_gens = train_dataset.num_gens    
    net= cnn.Net(species=num_specs, genuses=num_gens, num_channels=num_channels)
#     loss = torch.nn.BCELoss()
# multi loss from here: https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch 
    spec_loss = torch.nn.BCEWithLogitsLoss()
    gen_loss = torch.nn.BCEWithLogitsLoss()
#     fam_loss = torch.nn.BCEWithLogitsLoss()    
    optimizer = optim.Adam(net.parameters(), lr=ARGS.lr)
    model = net.to(device)
    
    
    def collate_fn(batch):
#         print("custom batch boi")
        # batch is a list of tuples of (specs_label, gens_label, fams_label, images)  
        all_specs = []
        all_gens = []
        all_fams = []
        imgs = []
        #(specs_label, gens_label, fams_label, images)  
        for (spec, gen, fam, img) in batch:
            specs_tens = torch.zeros(net.species)
            specs_tens[spec] += 1
            all_specs.append(specs_tens)

            gens_tens = torch.zeros(net.genuses)
            gens_tens[gen] += 1
            all_gens.append(gens_tens)

            fams_tens = torch.zeros(net.families)
            fams_tens[fam] += 1
            all_fams.append(fams_tens)
            imgs.append(img)
            
        return torch.stack(all_specs), torch.stack(all_gens), torch.stack(all_fams), torch.from_numpy(np.stack(imgs))



 

    train_loader = None
    test_loader = None
    test_dataset = None
    if ARGS.test:
        train_samp, test_samp = split_train_test(train_dataset, val_split)
        train_loader = DataLoader(train_dataset, ARGS.batch_size,  pin_memory=True, num_workers=ARGS.processes, sampler=train_samp, collate_fn=collate_fn) 
        test_loader = DataLoader(train_dataset, ARGS.batch_size,  pin_memory=True, num_workers=ARGS.processes, sampler=test_samp, collate_fn=collate_fn)
    else:
        train_loader = DataLoader(train_dataset, ARGS.batch_size, shuffle=True, pin_memory=True, num_workers=ARGS.processes, collate_fn=collate_fn) 
        if ARGS.country == 'both':
            test_dataset = Dataset.GEOCELF_Test_Dataset_Full(ARGS.base_dir)
            test_loader = DataLoader(test_dataset, ARGS.batch_size, shuffle=True, pin_memory=True, num_workers=ARGS.processes)
        else: 
            test_dataset = Dataset.GEOCELF_Test_Dataset(ARGS.base_dir, ARGS.country)
            test_loader = DataLoader(test_dataset, ARGS.batch_size, shuffle=True, pin_memory=True, num_workers=ARGS.processes)

    

    batch_size=ARGS.batch_size
    n_epochs=ARGS.epoch
    num_batches = math.ceil(len(train_dataset) / batch_size)
    print("batch size is {batch_size} and size of dataset is {lens} and number of batches is {num_batches}\n".format(batch_size=ARGS.batch_size, num_batches=len(train_loader), lens=len(train_dataset)))
    print("starting training") 
    all_time_loss = []
    all_time_sp_loss = []
    all_time_gen_loss = []
#     all_time_fam_loss = []    
    step = 0    
    for epoch in range(n_epochs):
        if ARGS.device is not None:
            torch.cuda.synchronize()
        tick = time.time()
        net.train()
        tot_loss_meter = []
        spec_loss_meter = []
        gen_loss_meter = []
#         fam_loss_meter = []  
        
        with tqdm(total=len(train_loader), unit="batch") as prog:

            for i, (specs_lab, gens_lab, fams_lab, batch) in enumerate(train_loader):
                batch = batch.to(device)
                specs_lab = specs_lab.to(device)                                     
                gens_lab = gens_lab.to(device)
#                 fams_lab = fams_lab.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                (specs, gens) = net(batch.float()) # convert to float so torch happy
                # size of specs: [N, species] gens: [N, genuses] fam: [N, fams]

                # compute loss https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch
                # https://stackoverflow.com/questions/48274929/pytorch-runtimeerror-trying-to-backward-through-the-graph-a-second-time-but
                loss_spec = spec_loss(specs, specs_lab) 
                loss_gen = gen_loss(gens, gens_lab) 
#                 loss_fam = fam_loss(fams, fams_lab)       
                total_loss = loss_spec + loss_gen #+ loss_fam
                total_loss.backward()
                optimizer.step()

                tot_loss = total_loss.item()
                tot_loss_meter.append(tot_loss)                
                spec_loss_meter.append(loss_spec.item())
                gen_loss_meter.append(loss_gen.item())
#                 fam_loss_meter.append( /loss_fam.item())                    
                prog.update(1)
                tb_writer.add_scalar("train/tot_loss", tot_loss, step)
                tb_writer.add_scalar("train/spec_loss", loss_spec.item(), step)
#                 tb_writer.add_scalar("train/fam_loss", loss_fam.item(), step)
                tb_writer.add_scalar("train/gen_loss", loss_gen.item(), step)   
                prog.set_description("loss: {tot_loss}".format(tot_loss=tot_loss))
                step += 1                
                # update loss tracker
        prog.close() 
        all_time_loss.append(np.stack(tot_loss_meter))
        all_time_sp_loss.append(np.stack(spec_loss_meter))
        all_time_gen_loss.append(np.stack(gen_loss_meter))
#         all_time_fam_loss.append(np.stack(fam_loss_meter))


        print ("Average Train Loss: {avg_loss}".format(avg_loss=np.stack(tot_loss_meter).mean(0)))
        del batch, specs_lab, gens_lab, specs, gens, loss_spec, loss_gen
        
        # save model 


        print("saving model for epoch {epoch}".format(epoch=epoch))
        PATH="{}nets/cnn_{}_{}.tar".format(ARGS.base_dir, ARGS.exp_id, epoch)
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'all_loss' : np.stack(all_time_loss),
                    'spec_loss': np.stack(all_time_sp_loss),
                    'gen_loss': np.stack(all_time_gen_loss), 
#                     'fam_loss': np.stack(all_time_fam_loss)
                    }, PATH)
        
        
        
        
        if ARGS.device is not None:
            print("cleaning gpu")
            torch.cuda.empty_cache()
        # test
        net.eval()
        # https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
        all_accs = []
        mean_accs = []
        with torch.no_grad():
            if ARGS.test:
                with tqdm(total=len(test_loader), unit="batch") as prog:
                    means = []
                    for i, (specs_label, gens_lab, fams_lab, loaded_imgs) in enumerate(test_loader):
                        tick = time.time()
                        batch = loaded_imgs.to(device)
                        specs_lab = specs_label.to(device)   
                        gens_lab = gens_lab.to(device)
                        (outputs, gens) = net(batch.float()) 
                        gen_accs, totgen_accs = topk_acc(gens, gens_lab) # magic no from CELF2020
                        spec_accs, totspec_accs = topk_acc(outputs, specs_lab) # magic no from CELF2020
                        prog.set_description("mean spec accuracy across batch: {acc0}".format(acc0=spec_accs.mean()))
                        prog.update(1)          
                        tb_writer.add_scalar("test/avg_spec_accuracy", spec_accs.mean(), epoch)
                        tb_writer.add_scalar("test/avg_gen_accuracy", gen_accs.mean(), epoch)                        
                        all_accs.append((tot_accs, totgen_accs))
                        mean_accs.append(spec_accs)
                        means.append(spec_accs.mean()) 
                prog.close()
                means = np.stack(means)
                print("max top 1 accuracy across batches: {max1} average top1 accuracy across batches: {avg1}".format(max1=means.max(), avg1=means.mean()))
                del outputs, specs_lab, batch, gens_lab, gens 
            else: 
                with tqdm(total=len(test_loader), unit="batch") as prog:
                    file = "{}output/{}_{}_e{}.csv".format(ARGS.base_dir, ARGS.country, ARGS.exp_id, epoch)
                    with open(file,'w') as f:
                        writer = csv.writer(f, dialect='unix')
                        top_class = ['top_{n}_class_id'.format(n=n) for n in np.arange(1, 151)]
                        top_score = ['top_{n}_class_score'.format(n=n) for n in np.arange(1, 151)]  
                        header = ['observation_id'] + top_class + top_score
                        writer.writerow(header)                        
                        for i, (batch, id_) in enumerate(test_loader):
                            batch = batch.to(device)                                  
                            (outputs, _, _) = net(batch.float()) 
                            scores, idxs = torch.topk(outputs.cpu(), dim=1, k=150)
                            top_scores = scores[:,:150]
                            top_idxs = idxs[:,:150]
                            for scores, ids, idd in zip(top_scores, top_idxs, id_):
                                ids = [train_dataset.idx_2_id[i.item()] for i in ids]
                                row = itertools.chain( [idd.item()], ids, scores.tolist())
                                writer.writerow(row)
                            prog.update(1)
                    print("saving to : {}".format(file))                            
                prog.close()
                f.close() 
                del batch, outputs
        if ARGS.device is not None:
            torch.cuda.empty_cache()


        tock = time.time()
        diff = ( tock-tick)/60
        print ("one epoch took {} minutes".format(diff))


if __name__ == "__main__":
    #print(f"torch version: {torch.__version__}") 
    #print(f"numpy version: {np.__version__}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate of model",required=True)
    parser.add_argument("--epoch", type=int, required=True, help="how many epochs to train the model")
    parser.add_argument("--device", type=int, help="which gpu to send model to, don't put anything to use cpu")
    parser.add_argument("--processes", type=int, help="how many worker processes to use for data loading", default=1)
    parser.add_argument("--exp_id", type=str, help="experiment id of this run", required=True)
    parser.add_argument("--base_dir", type=str, help="what folder to read images from",choices=['DBS_DIR', 'MNT_DIR', 'MEMEX_LUSTRE', 'CALC_SCRATCH', 'AZURE_DIR'], required=True)
    parser.add_argument("--country", type=str, help="which country's images to read", default='us', required=True, choices=['us', 'fr', 'both', 'cali'])
    parser.add_argument("--seed", type=int, help="random seed to use")
    parser.add_argument('--plants', dest='plants', help="if dataset used only looks at plantae occurrences", action='store_true')

    parser.add_argument('--test', dest='test', help="if set, split train into test, val set. If not seif set, split train into test, val set. If not set, train network on full dataset", action='store_true')
#     parser.add_argument("--load_size", type=int, help="how many instances to hold in memory at a time", default=1000)    
    parser.add_argument("--batch_size", type=int, help="size of batches to use", default=50)    
    ARGS, _ = parser.parse_known_args()
    # parsing which path to use
    ARGS.base_dir = eval("paths.{}".format(ARGS.base_dir))
    print("using base directory {}".format(ARGS.base_dir))
    # Seed
#     assert ARGS.load_size >= ARGS.batch_size, "load size must be bigger than batch size!"
    if ARGS.seed is not None:
        np.random.seed(ARGS.seed)
        torch.manual_seed(ARGS.seed)
    if not os.path.exists("{}output/".format(ARGS.base_dir)):
        os.makedirs("{}output/".format(ARGS.base_dir))
    if not os.path.exists("{}nets/".format(ARGS.base_dir)):
        os.makedirs("{}nets/".format(ARGS.base_dir))        
    main()
