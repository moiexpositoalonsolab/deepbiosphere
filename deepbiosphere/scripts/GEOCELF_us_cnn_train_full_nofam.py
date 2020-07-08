
from random import randrange
import pandas as pd
import os
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
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import random
import math
from tqdm import tqdm
from deepbiosphere.scripts import GEOCELF_CNN as cnn
from deepbiosphere.scripts import GEOCELF_Dataset as Dataset
from deepbiosphere.scripts import paths
        
# https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b        
def topk_acc(output, target, topk=(1,), device=None):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    targ = target.unsqueeze(1).repeat(1,maxk).to(device)
    correct = pred.eq(targ)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).cpu()
        res.append(correct_k.mul_(100.0 / batch_size))
    del targ, pred, target
    return res

def split_train_test(full_dat, split_amt):
    #grab split_amt% of labeled data for holdout testing
    idxs = np.random.permutation(len(full_dat))
    split = int(len(idxs)*split_amt)
    training_idx, test_idx = idxs[:split], idxs[split:]
    train_sampler = SubsetRandomSampler(training_idx)
    valid_sampler = SubsetRandomSampler(test_idx)
    return train_sampler, valid_sampler

def check_mem():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size(), obj.device)
        except: pass

def main():
    print("torch version {}".format(torch.__version__))
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
        train_dataset = Dataset.GEOCELF_Dataset_Full(ARGS.base_dir)
    else:
        split = 'plant' if ARGS.plants else ''        
        train_dataset = Dataset.GEOCELF_Dataset(ARGS.base_dir, ARGS.country, split)
    tb_writer = SummaryWriter(comment="exp_id: {}".format(ARGS.exp_id))
    val_split = .9
    train_loader = None
    test_loader = None
    test_dataset = None
        
        #TODO: implement GeoCLEF pipeline
    # set up net
    datock = time.time()
    dadiff = datock - datick
    print("loading data took {dadiff} seconds".format(dadiff=dadiff))
    print("setting up network")
    num_channels = train_dataset.channels# num_channels should be idx 1 in the order torch expects
    num_specs = train_dataset.num_specs
    num_fams = train_dataset.num_fams
    num_gens = train_dataset.num_gens    
    net= cnn.No_Fam_Net(species=num_specs, genuses=num_gens, num_channels=num_channels)
# multi loss from here: https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch 
    spec_loss = torch.nn.CrossEntropyLoss()
    gen_loss = torch.nn.CrossEntropyLoss()
#     fam_loss = torch.nn.CrossEntropyLoss()    
    optimizer = optim.Adam(net.parameters(), lr=ARGS.lr)

    net.to(device)    
    def collate_fn(batch): 
        # batch is a list of tuples of (composite_label <np array [3]>, images <np array [6, 256, 256]>)   
        labs, img = zip(*batch) 
        lbs = [torch.tensor(l, dtype=torch.long) for l in labs]      
        img = [i.astype(np.uint8, copy=False) for i in img]
        imgs = [torch.from_numpy(i) for i in img]
        return torch.stack(lbs), torch.stack(imgs)    

    if ARGS.test:
        train_samp, test_samp = split_train_test(train_dataset, val_split)
        train_loader = DataLoader(train_dataset, ARGS.batch_size,  pin_memory=True, num_workers=ARGS.processes, sampler=train_samp, collate_fn=collate_fn) 
        test_loader = DataLoader(train_dataset, ARGS.batch_size,  pin_memory=True, num_workers=ARGS.processes, sampler=test_samp, collate_fn=collate_fn)
        test_dataset = train_dataset
    else:
#         train_loader = DataLoader(train_dataset, ARGS.load_size, shuffle=True, pin_memory=False, num_workers=ARGS.processes) 
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
    print("batch size is {batch_size} and size of dataset is {lens} and num batches is {num_batches}\n".format(batch_size=ARGS.batch_size, lens=len(train_dataset), num_batches=len(train_loader)))
    print("starting training") 
    all_time_loss = []
    all_time_sp_loss = []
    all_time_gen_loss = []
#     all_time_fam_loss = []  
    step = 0
    for epoch in range(n_epochs):
        print("starting training for epoch {}".format(epoch))

        if ARGS.device is not None:
            torch.cuda.synchronize()
        tick = time.time()
        net.train()
        tot_loss_meter = []
        spec_loss_meter = []
        gen_loss_meter = []
#         fam_loss_meter = []        

        with tqdm(total=len(train_loader), unit="batch") as prog:
            
            for i, (labels, batch) in enumerate(train_loader):
                # Loop inside loaded data

                batch = batch.to(device)
                labels = labels.to(device)                             
                # zero the parameter gradients
#                 tuck = time.time()
                optimizer.zero_grad()
                # forward + backward + optimize
                (specs, gens) = net(batch.float()) # convert to float so torch happy
                # size of specs: [N, species] gens: [N, genuses] fam: [N, fams]

                # compute loss https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch
                # https://stackoverflow.com/questions/48274929/pytorch-runtimeerror-trying-to-backward-through-the-graph-a-second-time-but
                # https://stackoverflow.com/questions/46774641/what-does-the-parameter-retain-graph-mean-in-the-variables-backward-method
                loss_spec = spec_loss(specs, labels[:,0]) 
                loss_gen = gen_loss(gens, labels[:,1]) 
#                 loss_fam = fam_loss(fams, labels[:,2])
                total_loss = loss_spec + loss_gen# + loss_fam
                total_loss.backward()
                optimizer.step()
                # optimizer.zero_grad()
                # update tqdm

                tot_loss = total_loss.item()
                tot_loss_meter.append(tot_loss)                
                spec_loss_meter.append(loss_spec.item())
                gen_loss_meter.append(loss_gen.item())
#                 fam_loss_meter.append(loss_fam.item())                    
                prog.update(1)
                tb_writer.add_scalar("train/tot_loss", tot_loss, step)
                tb_writer.add_scalar("train/spec_loss", loss_spec.item(), step)
#                 tb_writer.add_scalar("train/fam_loss", loss_fam.item(), step)
                tb_writer.add_scalar("train/gen_loss", loss_gen.item(), step)                
                step += 1
                prog.set_description("loss: {tot_loss}".format(tot_loss=tot_loss))
            # update loss tracker
        prog.close() 
        all_time_loss.append(np.stack(tot_loss_meter))
        all_time_sp_loss.append(np.stack(spec_loss_meter))
        all_time_gen_loss.append(np.stack(gen_loss_meter))
#         all_time_fam_loss.append(np.stack(fam_loss_meter))

        
        
        print ("Average Train Loss: {avg_loss}".format(avg_loss=np.stack(tot_loss_meter).mean(0)))

        del batch, labels, specs, gens, loss_spec, loss_gen# loss_fam
        
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
        print("testing model")
        with torch.no_grad():
            if ARGS.test:
                with tqdm(total=len(test_loader), unit="batch") as prog:
                    for i, (labels, batch) in enumerate(test_loader):

                        labels = specs_lab.to(device)
                        batch = batch.to(device)

                        (outputs, gens) = net(batch.float()) 
                        spec_accs = topk_acc(outputs, labels[:,0], topk=(30,1), device=device) # magic no from CELF2020
                        gen_accs = topk_acc(gens, labels[:,1], topk=(30,1), device=device) # magic no from CELF2020                        
                        prog.set_description("top 30: {acc0}  top1: {acc1}".format(acc0=spec_accs[0], acc1=spec_accs[1]))
                        all_accs.append(spec_accs)
                        tb_writer.add_scalar("test/30_spec_accuracy", spec_accs[0], epoch)
                        tb_writer.add_scalar("test/1_spec_accuracy", spec_accs[1], epoch) 
                        
                        tb_writer.add_scalar("test/30_gen_accuracy", gen_accs[0], epoch)
                        tb_writer.add_scalar("test/1_gen_accuracy", gen_accs[1], epoch)                         
                        prog.update(1)
                prog.close()
                all_accs = np.stack(all_accs)
                print("max top 30 accuracy: {max1} average top1 accuracy: {max2}".format(max1=all_accs[:,0].max(), max2=all_accs[:,1].max()))
                del outputs, labels, batch 
            else:
                with tqdm(total=len(test_loader), unit="obs") as prog:
                    file = "{}output/{}_{}_e{}.csv".format(ARGS.base_dir, ARGS.country, ARGS.exp_id, epoch)
                    with open(file,'w') as f:
                        writer = csv.writer(f, dialect='unix')
                        top_class = ['top_{n}_class_id'.format(n=n) for n in np.arange(1, 151)]
                        top_score = ['top_{n}_class_score'.format(n=n) for n in np.arange(1, 151)]  
                        header = ['observation_id'] + top_class + top_score
                        writer.writerow(header)
                        for i, (batch, id_) in enumerate(test_loader):
                            batch = batch.to(device)                                  
                            (outputs, _) = net(batch.float()) 
                            scores, idxs = torch.topk(outputs.cpu(), dim=1, k=150)
                            top_scores = scores[:,:150]
                            top_idxs = idxs[:,:150]
                            for scores, ids, idd in zip(top_scores, top_idxs, id_):
                                ids = [train_dataset.idx_2_id[i.item()] for i in ids]
                                row = itertools.chain( [idd.item()], ids, scores.tolist())
                                writer.writerow(row)
                            prog.update(1)
                prog.close()
                del outputs, batch                 

        if ARGS.device is not None:
            torch.cuda.empty_cache()
      
    


        tock = time.time()
        diff = ( tock-tick)/60
        print ("one epoch took {} minutes".format(diff))
    tb_writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate of model",required=True)
    parser.add_argument("--epoch", type=int, required=True, help="how many epochs to train the model")
    parser.add_argument("--device", type=int, help="which gpu to send model to, don't put anything to use cpu")
    parser.add_argument("--processes", type=int, help="how many worker processes to use for data loading", default=1)
    parser.add_argument("--exp_id", type=str, help="experiment id of this run", required=True)
    parser.add_argument("--base_dir", type=str, help="what folder to read images from",choices=['DBS_DIR', 'MNT_DIR', 'MEMEX_LUSTRE', 'CALC_SCRATCH', 'AZURE_DIR'], required=True)
    parser.add_argument("--country", type=str, help="which country's images to read", default='us', required=True, choices=['us', 'fr', 'both', 'cali'])
    parser.add_argument('--plants', dest='plants', help="if dataset used only looks at plantae occurrences", action='store_true')
    parser.add_argument("--seed", type=int, help="random seed to use")
    parser.add_argument('--test', dest='test', help="if set, split train into test, val set. If not seif set, split train into test, val set. If not set, train network on full dataset", action='store_true')
    parser.add_argument("--load_size", type=int, help="how many instances to hold in memory at a time", default=1000)    
    parser.add_argument("--batch_size", type=int, help="size of batches to use", default=50)    
    ARGS, _ = parser.parse_known_args()
    # parsing which path to use
    ARGS.base_dir = eval("paths.{}".format(ARGS.base_dir))
    print("using base directory {}".format(ARGS.base_dir))
    # Seed
    assert ARGS.load_size >= ARGS.batch_size, "load size must be bigger than batch size!"
    if ARGS.seed is not None:
        np.random.seed(ARGS.seed)
        torch.manual_seed(ARGS.seed)
    if not os.path.exists("{}output/".format(ARGS.base_dir)):
        os.makedirs("{}output/".format(ARGS.base_dir))
    if not os.path.exists("{}nets/".format(ARGS.base_dir)):
        os.makedirs("{}nets/".format(ARGS.base_dir))        
    main()
