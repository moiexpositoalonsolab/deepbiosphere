from random import randrange
import pandas as pd
import argparse
import time
import numpy as np
import socket
import torch
import torch.nn as nn
import torch.optim as optim
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

def main():
    # gg = "gg{ff}".format(ff=ff)
    print("number of devices visible: {dev}".format(dev=torch.cuda.device_count()))
    device = torch.device("cuda:{dev}".format(dev=ARGS.device) if ARGS.device is not None else "cpu")
    print('using device: {device}'.format(device=device))
    if ARGS.device is not None:
        print("current device: {dev} current device name: {name}".format(dev=torch.cuda.current_device(), name=torch.cuda.get_device_name(torch.cuda.current_device())))
    print("current host: {host}".format(host=socket.gethostname()))

    # load observation data
    print("loading data")
    datick = time.time()
    train_dataset = Dataset.GEOCELF_Dataset(ARGS.base_dir, 'train', ARGS.country)
    val_split = .9
    train_loader = None
    test_loader = None
    test_dataset = None
    if ARGS.test:
        train_samp, test_samp = split_train_test(train_dataset, val_split)
        train_loader = DataLoader(train_dataset, ARGS.batch_size,  pin_memory=True, num_workers=ARGS.processes, sampler=train_samp) 
        test_loader = DataLoader(train_dataset, ARGS.batch_size,  pin_memory=True, num_workers=ARGS.processes, sampler=test_samp)
    else:
        train_loader = DataLoader(train_dataset, ARGS.batch_size, shuffle=True, pin_memory=True, num_workers=ARGS.processes) 
        test_dataset = Dataset.GEOCELF_Test_Dataset(ARGS.base_dir, 'test', ARGS.country)
        test_loader = DataLoader(test_dataset, ARGS.batch_size, shuffle=True, pin_memory=True, num_workers=ARGS.processes)
    # set up net
    datock = time.time()
    dadiff = datock - datick
    print("loading data took {dadiff} seconds".format(dadiff=dadiff))
    print("setting up network")
    num_channels = train_dataset.channels# num_channels should be idx 1 in the order torch expects
    num_specs = train_dataset.num_specs
    num_fams = train_dataset.num_fams
    num_gens = train_dataset.num_gens    
    net= cnn.Net(species=num_specs, families=num_fams, genuses=num_gens, num_channels=num_channels)
#     loss = torch.nn.BCELoss()
# multi loss from here: https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch 
    spec_loss = torch.nn.CrossEntropyLoss()
    gen_loss = torch.nn.CrossEntropyLoss()
    fam_loss = torch.nn.CrossEntropyLoss()    
    optimizer = optim.Adam(net.parameters(), lr=ARGS.lr)
    model = net.to(device)

    


    batch_size=ARGS.batch_size
    n_epochs=ARGS.epoch
    num_batches = math.ceil(len(train_dataset) / batch_size)
    print("batch size is {batch_size} and size of dataset is {lens} total size of dataset is {len_dat} and num batches is {num_batches}\n".format(batch_size=batch_size, lens=len(train_loader), len_dat=len(train_dataset), num_batches=num_batches))
    print("starting training") 
    all_time_loss = []
    all_time_sp_loss = []
    all_time_gen_loss = []
    all_time_fam_loss = []    
    for epoch in range(n_epochs):
        if ARGS.device is not None:
            torch.cuda.synchronize()
        tick = time.time()
        net.train()
        tot_loss_meter = []
        spec_loss_meter = []
        gen_loss_meter = []
        fam_loss_meter = []        
        with tqdm(total=len(train_loader), unit="batch") as prog:
            for i, (labels, batch) in enumerate(train_loader):
                batch = batch.to(device)
                labels = labels.to(device)                                     
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                (specs, gens, fams) = net(batch.float()) # convert to float so torch happy
                # size of specs: [N, species] gens: [N, genuses] fam: [N, fams]
#                  for BCELoss
#                 spec_labels = F.one_hot(specs, net.species)
#                 gen_labels = F.one_hot(gens, net.genuses)
#                 fam_labels = F.one_hot(fams, net.families)
                
                # compute loss https://stackoverflow.com/questions/53994625/how-can-i-process-multi-loss-in-pytorch
                # https://stackoverflow.com/questions/48274929/pytorch-runtimeerror-trying-to-backward-through-the-graph-a-second-time-but
                loss_spec = spec_loss(specs, labels[:,0]) 
                loss_spec.backward(retain_graph=True)
                loss_gen = gen_loss(gens, labels[:,1]) 
                loss_gen.backward(retain_graph=True)
                loss_fam = fam_loss(fams, labels[:,2])                 
                loss_fam.backward()
                optimizer.step()
                # update tqdm

                tot_loss = loss_spec.item() + loss_gen.item() + loss_fam.item()
                tot_loss_meter.append(tot_loss)                
                spec_loss_meter.append(loss_spec.item())
                gen_loss_meter.append(loss_gen.item())
                fam_loss_meter.append(loss_fam.item())                    
                prog.update(1)
                prog.set_description("loss: {tot_loss}".format(tot_loss=tot_loss))
                # update loss tracker
        prog.close() 
        all_time_loss.append(np.stack(tot_loss_meter))
        all_time_sp_loss.append(np.stack(spec_loss_meter))
        all_time_gen_loss.append(np.stack(gen_loss_meter))
        all_time_fam_loss.append(np.stack(fam_loss_meter))

        
        
        print ("Average Train Loss: {avg_loss}".format(avg_loss=np.stack(tot_loss_meter).mean(0)))
#         all_time_loss.append(np.stack(loss_meter))
        del batch, labels, specs, gens, fams, loss_spec, loss_gen, loss_fam
        if ARGS.device is not None:
            print("cleaning gpu")
            torch.cuda.empty_cache()
        # test
        net.eval()
        # https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
        with torch.no_grad():

#             idxs = np.random.permutation(len(test_dataset))
#             labels, batch = test_dataset[idxs[:batch_size]]
#             labels, batch = test_dataset[randrange(len(test_loader)-batch_size)]
            all_accs = []
            with tqdm(total=len(test_loader), unit="batch") as prog:
                for i, (labels, batch) in enumerate(test_loader):
                    labels = labels.to(device)
                    batch = batch.to(device)

                    (outputs, _, _) = net(batch.float()) 
                    if ARGS.test:
                        accs = topk_acc(outputs, labels[:,0], topk=(30,1), device=device) # magic no from CELF2020
                        prog.set_description("top 30: {acc0}  top1: {acc1}".format(acc0=accs[0], acc1=accs[1]))
                        all_accs.append(accs)
                    else:
                        all_accs.append(outputs.cpu())
                prog.close()
                all_accs = np.stack(all_accs)
                print("max top 30 accuracy: {max1} average top1 accuracy: {max2}".format(max1=all_accs[:,0].max(), max2=all_accs[:,1].max()))
                del outputs, labels, batch 
            if ARGS.device is not None:
                torch.cuda.empty_cache()
      

        # save model 


        print("saving model for epoch {epoch}".format(epoch=epoch))
        PATH="{}cnn_{}.tar".format(paths.NETS_DIR, ARGS.exp_id)
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'all_loss' : np.stack(all_time_loss),
                    'spec_loss': np.stack(all_time_sp_loss),
                    'gen_loss': np.stack(all_time_gen_loss), 
                    'fam_loss': np.stack(all_time_fam_loss), 
                    'accuracy': all_accs,
                    }, PATH)

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
    parser.add_argument("--base_dir", type=str, help="what folder to read images from",choices=['DBS_DIR', 'MEMEX_LUSTRE', 'CALC_SCRATCH', 'AZURE_DIR'], required=True)
    parser.add_argument("--country", type=str, help="which country's images to read", default='us')
    parser.add_argument("--seed", type=int, help="random seed to use")
    parser.add_argument('--test', dest='test', help="if set, split train into test, val set. If not seif set, split train into test, val set. If not set, train network on full datasett", action='store_true')
    parser.add_argument("--batch_size", type=int, help="size of batches to use", default=256)    
    ARGS, _ = parser.parse_known_args()
    # parsing which path to use
    ARGS.base_dir = eval("paths.{}".format(ARGS.base_dir))
    print("using base directory {}".format(ARGS.base_dir))
    # Seed
    if ARGS.seed is not None:
        np.random.seed(ARGS.seed)
        torch.manual_seed(ARGS.seed)
    main()
