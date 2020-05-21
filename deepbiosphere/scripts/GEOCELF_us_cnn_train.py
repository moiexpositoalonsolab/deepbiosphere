import pandas as pd
import argparse
import time
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import math
from tqdm import tqdm
from deepbiosphere.scripts import GEOCELF_CNN as cnn
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
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    del targ
    return res

def us_image_from_id(id_, pth):
    abcd = id_ % 10000
    ab, cd = math.floor(abcd/100), abcd%100
    cdd = math.ceil((cd+ 1)/5)
    cdd = f"0{cdd}"  if cdd < 10 else f"{cdd}"
    ab = f"0{ab}" if id_ / 1000 > 1 and ab < 10 else ab
    cd = f"0{cd}" if id_ / 1000 > 1 and cd < 10 else cd
    subpath = f"patches_us_{cdd}/{cd}/{ab}/"
    alt = f"{pth}{subpath}{id_}_alti.npy"
    rgbd = f"{pth}{subpath}{id_}.npy"    
    np_al = np.load(alt)
    np_img = np.load(rgbd)
    np_al = np.expand_dims(np_al, 2)
    np_all = np.concatenate((np_al, np_img), axis=2)
    return np_all

def tensor_from_ids(ids, img_pth, device):
    x_tens = [us_image_from_id(id_, img_pth) for id_ in ids]
    x_tens = torch.from_numpy(np.stack(x_tens)) #should be size 256 x 256 x 6 x batchsize
    x_tens = x_tens.permute(0,3,1,2)# reshape to shape torch expects: ((N,Cin,H,W))
    return x_tens

def get_gbif_data(pth, country):
    ## Grab GBIF observation data
    train_pth = f"{pth}occurrences/occurrences_{country}_train.csv"
    test_pth = f"{pth}occurrences/occurrences_{country}_test.csv"
    train = pd.read_csv(train_pth, sep=';')
    test = pd.read_csv(test_pth, sep=';')  
    return train, test
  
    
def split_train_test(full_dat, split_amt):
    #grab 20% of labeled data for holdout testing
    idxs = np.random.permutation(len(full_dat))
    split = int(len(idxs)*split_amt)
    training_idx, test_idx = idxs[:split], idxs[split:]
    training, test = full_dat[training_idx,:], full_dat[test_idx,:]  
    assert not np.array_equal(training[:20], full_dat[:20]), "permutation didn't work!"
    return training, test

def split_id_specs(obs_data, device):
    obs_ids, obs_spec_ids = obs_data[:,0], obs_data[:,1]
    obs_spec_ids = torch.from_numpy(obs_spec_ids)
    obs_spec_ids = obs_spec_ids
    return obs_ids, obs_spec_ids

def shuffle_data(obs, labels):
    indxs = torch.randperm(len(labels))
    return  obs[indxs], labels[indxs]
    

def batch_data(obs, labels, batch_size):
#     assert training_imgs.shape[0] == len(train_ids), "number of training examples and labels don't match!"
    curr_obs, curr_labels = shuffle_data(obs, labels)
    label_bat = torch.split(curr_labels, batch_size)
    obs_bat = torch.split(curr_obs, batch_size)
    return obs_bat, label_bat

def main():
    
    device = torch.device(f"cuda:{ARGS.device}" if ARGS.device is not None else "cpu")
    print(f'using device: {device}')
    
    # load observation data
    print("loading labels")
    pth = paths.GEOCELF_DIR

    us_train, _ = get_gbif_data(pth, 'us')
    spec_2_id = {k:v for k, v in zip(us_train.species_id.unique(), np.arange(len(us_train.species_id.unique())))}
    us_train['species_id'] = us_train['species_id'].map(spec_2_id)
    assert us_train['species_id'].max()+1 == len(us_train.species_id.unique()), f"map unsuccessful. {us_train['species_id'].max()} vs {len(us_train.species_id.unique())}"
    # Grab only obs id, species id because lat /lon not necessary at the moment
    us_train = us_train[['id', 'species_id']]
    print("labels loaded")

    #grab 20% of labeled data for holdout testing
    training, test = split_train_test(us_train.to_numpy(), .8)
    train_ids, train_spec_ids = split_id_specs(training, device)
    test_ids, test_spec_ids = split_id_specs(test, device)
    
    print("loading images")    
    tick = time.time()
    training_imgs = tensor_from_ids(train_ids, f"{paths.GEOCELF_DIR}patches_us/", ARGS.device)
    test_imgs = tensor_from_ids(test_ids, f"{paths.GEOCELF_DIR}patches_us/", ARGS.device)
    tock = time.time()
    diff = tock - tick
    print(f"images loaded. Took {diff} seconds")
    
    # set up net
    num_channels = training_imgs.shape[1]# num_channels should be idx 1 in the order torch expects
    num_cats = len(us_train.species_id.unique())
    net= cnn.Net(categories=num_cats, num_channels=num_channels)
    loss = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=ARGS.lr)
    model = net.to(device)

    


    batch_size=ARGS.batch_size
    n_epochs=ARGS.epoch
    n_minibatches = math.ceil(len(train_ids) / batch_size)

    
    for epoch in range(n_epochs):
        net.train()
        loss_meter = []
        with tqdm(total=(n_minibatches)) as prog:
            
            train_bat, label_bat = batch_data(training_imgs, train_spec_ids, batch_size)

        #             assert len(label_bat) == len(train_bat), f"input: {len(label_bat)}, label: {len(train_bat)} batches aren't sized correctly!"
#             assert label_bat[-1].shape[0] == train_bat[-1].shape[0], "number of training examples and labels don't match!"
            for batch, labels in zip(train_bat, label_bat):
                batch = batch.to(device)
                labels = labels.to(device)                                     
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = net(batch.float()) # convert to float so torch happy
#                 assert outputs.shape[0] == len(labels); f"your logits {outputs.shape} and label {len(labels)} sizes do not match up!"
                # compute loss
                loss_rec = loss(outputs, labels) 
                loss_rec.backward()
                optimizer.step()
                # update tqdm
                prog.update(1)
                # update loss tracker
                loss_meter.append(loss_rec.item())                                     
                
        print (f"Average Train Loss: {np.stack(loss_meter).mean(0)}")
        # test

        #TODO: add batching to eval loop
        net.eval()
        curr_test, curr_labels = shuffle_data(test_imgs, test_spec_ids)
        curr_test = curr_test[:20]
        test_labels = curr_labels[:20]
        curr_test = curr_test.to(device)
        
        outputs = net(curr_test.float()) 
        accs = topk_acc(outputs, test_labels, topk=(30,1), device=device) # magic no from CELF2020
        print(f"average top 30 accuracy: {accs[0]} average top1 accuracy: {accs[1]}")
        del outputs
        
        # save model 
        print(f"saving model for epoch {epoch}")
        PATH=f"{paths.NETS_DIR}cnn_{ARGS.exp_id}.tar"
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss_rec,
                    'accuracy': accs
                    }, PATH)



if __name__ == "__main__":
    #print(f"torch version: {torch.__version__}") 
    #print(f"numpy version: {np.__version__}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate of model",required=True)
    parser.add_argument("--epoch", type=int, help="how many epochs to train the model")
    parser.add_argument("--device", type=int, help="which gpu to send model to, don't put anything to use cpu")
    parser.add_argument("--exp_id", type=str, help="experiment id of this run", required=True)
    parser.add_argument("--seed", type=int, help="random seed to use")
    parser.add_argument("--batch_size", type=int, help="size of batches to use", default=256)    
    ARGS, _ = parser.parse_known_args()
    # Seed
    if ARGS.seed is not None:
        np.random.seed(ARGS.seed)
        torch.manual_seed(ARGS.seed)
    main()
