
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



def main():
    device = torch.device("cuda:{dev}".format(dev=ARGS.device) if ARGS.device is not None else "cpu")    
    test_loader = None
    test_dataset = None
    
    
    if ARGS.country == 'both':
        train_dataset = Dataset.GEOCELF_Dataset_Full(ARGS.base_dir)
    else:
        train_dataset = Dataset.GEOCELF_Dataset(ARGS.base_dir, ARGS.country)    
    
    if ARGS.country == 'both':
        test_dataset = Dataset.GEOCELF_Test_Dataset_Full(ARGS.base_dir)
        test_loader = DataLoader(test_dataset,batch_size =1, shuffle=True, pin_memory=True, num_workers=ARGS.processes)
    else: 
        test_dataset = Dataset.GEOCELF_Test_Dataset(ARGS.base_dir, ARGS.country)
        test_loader = DataLoader(test_dataset,batch_size =1, shuffle=True, pin_memory=True, num_workers=ARGS.processes)
#     PATH="{}nets/cnn_{}.tar".format(paths.DBS_DIR, ARGS.exp_id)
    PATH="{}nets/cnn_{}_{}.tar".format(paths.DBS_DIR, ARGS.exp_id, epoch)
    net_dict = torch.load(PATH)
    epoch = net_dict['epoch']
    num_channels = train_dataset.channels# num_channels should be idx 1 in the order torch expects
    num_specs = train_dataset.num_specs
    num_fams = train_dataset.num_fams
    num_gens = train_dataset.num_gens    
    net= cnn.Net(species=num_specs, families=num_fams, genuses=num_gens, num_channels=num_channels)

    net.load_state_dict(net_dict['model_state_dict'])
    net.eval()
    net.to(device)
    with torch.no_grad():
        with tqdm(total=len(test_loader), unit="batch") as prog:
            file = "{}output/{}_{}_e{}{}.csv".format(ARGS.base_dir, ARGS.country, ARGS.exp_id, epoch, ARGS.identifier)
            with open(file,'w') as f:
                writer = csv.writer(f, dialect='excel')
                header = ['observation_id'] + ['top_class_id'] * 150 + ['top_class_score'] * 150
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



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=int, help="which gpu to send model to, don't put anything to use cpu")
    parser.add_argument("--n_spec", type=int, help="number of species trained with this model")
    parser.add_argument("--n_gen", type=int, help="number of genuses trained with this model") 
    parser.add_argument("--n_fam", type=int, help="number of families trained with this model")    
    parser.add_argument("--n_chan", type=int, help="number of channels trained with this model")    
#     parser.add_argument("--epoch", type=int, help="which epoch to load model from", required=True)    
    parser.add_argument("--exp_id", type=str, help="experiment id of this run", required=True)
    parser.add_argument("--identifier", type=str, help="extra identifier to add")    
    parser.add_argument("--base_dir", type=str, help="what folder to read images from",choices=['DBS_DIR', 'MEMEX_LUSTRE', 'CALC_SCRATCH', 'AZURE_DIR'], required=True)
    parser.add_argument("--processes", type=int, help="how many worker processes to use for data loading", default=0)
    parser.add_argument("--country", type=str, help="which country's images to read", default='us', required=True, choices=['us', 'fr', 'both'])
    ARGS, _ = parser.parse_known_args()
    # parsing which path to use
    ARGS.base_dir = eval("paths.{}".format(ARGS.base_dir))
    print("using base directory {}".format(ARGS.base_dir))
    main()