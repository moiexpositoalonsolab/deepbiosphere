# deepbiosphere packages
from deepbiosphere.Models import Model as mods
import deepbiosphere.Utils as utils
from deepbiosphere.Utils import paths
from deepbiosphere.Losses import Loss as losses
import deepbiosphere.Dataset as dataset
import deepbiosphere.TResNet as tresnet

# torch packages
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF
import sklearn.metrics as mets

# statistics packages
import numpy as np
import scipy.stats
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import precision_score, recall_score, f1_score, label_ranking_average_precision_score
import sklearn.metrics as mets
from numpy.random import default_rng
# trying to make random number gernerator more reproducible
# https://stackoverflow.com/questions/62309424/does-numpy-random-seed-always-give-the-same-random-number-every-time

# miscellaneous packages
import os
import time
import json
import random
import argparse
from tqdm import tqdm
from datetime import date
from csv import DictWriter
from types import SimpleNamespace

## ---------- Dataset and Dataloader fns ---------- ##

# need to custom-define our collate function since
# our dataset function returns a list of values
# turns out np.stack, torch from_numpy is faster than torch.stack
def collate(batch):
    # gotta handle the case where both bioclim and NAIP images are used
    
    if len(batch[0][3]) == 2:
        all_specs, all_gens, all_fams, imgras = zip(*batch)
        (images, rasters) = zip(*imgras)
        return torch.stack(all_specs),torch.stack(all_gens),torch.stack(all_fams),(torch.stack(images), torch.stack(rasters))
    # standard image or bioclim only training
    else:
        all_specs, all_gens, all_fams, images = zip(*batch)
        return torch.stack(all_specs),torch.stack(all_gens),torch.stack(all_fams),torch.stack(images)

# clunky but deals with separate return types
def batch_to_gpu(batch, device, args):
    
    # get contents
    spec_true, gen_true, fam_true, inputs = batch
    spec_true = spec_true.to(device)
    gen_true = gen_true.to(device)
    fam_true = fam_true.to(device)
    # to handle joint model
    if dataset.DataType[args.datatype] is dataset.DataType.JOINT_NAIP_BIOCLIM:
        # put inputs on gpu (hacky)
        inputs = (inputs[0].float().to(device), inputs[1].float().to(device))
    else:
        inputs = inputs.float().to(device)
    return spec_true, gen_true, fam_true, inputs
    
## ---------- tensorboard logging helper fns ---------- ##

def get_deltaP(x,y):

    # pres and absent classes
    targets = y
    anti_targets = 1 - y

    # Calculating Probabilities
    xs_pos = torch.sigmoid(x)
    xs_neg = 1.0 - xs_pos
    pos_prob = (targets * xs_pos).mean()
    neg_prob = (anti_targets * xs_neg).mean()
    return pos_prob.item() - neg_prob.item()

def record_deltaP(x,y, tb_writer, step, split, taxon):
    dp = get_deltaP(x,y)
    tb_writer.add_scalar(f"{split}/{taxon}_deltaP", dp, step)
    
    
## ---------- Config / data storage helper fns ---------- ##

def make_config(args, train_dset, shared_species):
    # save add'tl dataset attributes to config
    # hacky but handle taxonomic resolution
    args.nspecs = train_dset.nspec if 'spec' in args.taxon_type else -1
    args.ngens = train_dset.ngen if 'gen' in args.taxon_type else -1
    args.nfams = train_dset.nfam if 'fam' in args.taxon_type else -1
    args.nrasters = train_dset.nrasters
    args.shared_species = shared_species
    # TODO: change to match pretraining (if applicable)
    args.image_stats = f'naip_{args.year}'
        
    # save arguments
    path = f"{paths.MODELS}configs/{args.model}_{args.loss}_band{args.band}_{args.exp_id}.json"
    # if this is the first time building the config directory
    # make the directories and save out to disk
    currdir = os.path.dirname(path)
    if not os.path.exists(currdir):
        os.makedirs(currdir)
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=4)
    return args
        

def save_model(model, optimizer, epoch, args, steps):
        model_path= f"{paths.MODELS}{args.model}_{args.loss}/{args.exp_id}_lr{str(args.lr).split('.')[-1]}_e{epoch}.tar"
        # if this is the first time building the model directory
        # make the directories and save out to disk
        currdir = os.path.dirname(model_path)
        if not os.path.exists(currdir):
            os.makedirs(currdir)
        torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'step' : steps,
                    }, model_path)

def write_traintime(total, args, name, device):
        towrite = {
            'time (in hours)' : total,
            'date' : date.today(),
            'code' : 'python',
            'model' : args.model,
            'dataset' : args.dataset_name,
            'band' : args.band,
            'gpu_mem' : torch.cuda.memory_allocated(device),
            'epochs' : args.epochs,
            'exp_id' : args.exp_id
        }
        # save how long it took to run
        headers = ['time (in hours)', 'date', 'code', 'model', 'dataset', 'band', 'gpu_mem' , 'epochs', 'exp_id']
        fname = f"{paths.MODELS}timing_runs_{name}.csv"
        fexists = os.path.isfile(fname)
        with open(fname, 'a', newline='\n') as f:
            dctw = DictWriter(f, fieldnames=headers)
            # Pass the data in the dictionary as an argument into the writerow() function
            if not fexists:
                dctw.writeheader() # write header if not yet written
            dctw.writerow(towrite)        

## ---------- Instantiating pipeline components ---------- ##
    
def instantiate_datasets(cfg):
    if cfg.all_points:
        dset = dataset.DeepbioDataset(cfg.dataset_name, cfg.datatype, cfg.dataset_type, cfg.state, cfg.year, cfg.band, 'all_points', cfg.augment)
        specs = dset.pres_specs
        return dset, specs
    else:
        test_dset = dataset.DeepbioDataset(cfg.dataset_name, cfg.datatype, cfg.dataset_type, cfg.state, cfg.year, cfg.band, 'test', 'NONE')
        train_dset = dataset.DeepbioDataset(cfg.dataset_name, cfg.datatype, cfg.dataset_type, cfg.state, cfg.year, cfg.band, 'train', cfg.augment)
        # figure out what species are present in both the train and the test split
        # and only calculate accuracy metrics for those shared species
        shared_species = list(set(train_dset.pres_specs) & set(test_dset.pres_specs))
        return train_dset, test_dset, shared_species


# grab the right kind of model for the configuration
def instantiate_model(device, cfg):
    # typecheck model
    model_type = mods[cfg.model]
    if model_type is mods.RS_TRESNET:
        model = mods.RS_TRESNET(cfg.nspecs, 
                                 cfg.ngens, 
                                 cfg.nfams, 
                                 cfg.pretrain, 
                                 paths.MODELS)
    elif model_type is mods.BIOCLIM_MLP:
        model = mods.BIOCLIM_MLP(cfg.nspecs, 
                                       cfg.ngens, 
                                       cfg.nfams, 
                                       cfg.nrasters)
    elif model_type is mods.INCEPTION:
        model = mods.INCEPTION(cfg.nspecs)
    elif model_type is mods.DEEPBIOSPHERE:
        model = mods.DEEPBIOSPHERE(cfg.nspecs, 
                                         cfg.ngens, 
                                         cfg.nfams, 
                                         cfg.nrasters, 
                                         cfg.pretrain, 
                                         paths.MODELS)
    model = model.to(device)
    return model


def instantiate_loss(cfg, dset, device):

    # type check loss
    loss_type = losses[cfg.loss]
    # add weighting for weighted versions
    if (loss_type is losses.WEIGHTED_CE) or (loss_type is losses.WEIGHTED_BCE):
        return loss_type(dset.metadata.species_counts,dset.metadata.spec_2_id, dset.total_len, device)
    else:
        return loss_type()
    
## ---------- re-loading previous experiments ---------- ##
    
def load_config(exp_id, band, loss, model):
    path = f"{paths.MODELS}configs/{model}_{loss}_band{band}_{exp_id}.json"
    with open(path, 'r') as f:
        return SimpleNamespace(**json.load(f))

def load_model(device, cfg, epoch, eval_=True, logging=True):
    if logging:
        print(f"Loading model {cfg.model} trained on band {cfg.band} with loss {cfg.loss}")
    model_path= f"{paths.MODELS}{cfg.model}_{cfg.loss}/{cfg.exp_id}_lr{str(cfg.lr).split('.')[-1]}_e{epoch}.tar"
    mdict =  torch.load(model_path, map_location=device)
    # TODO: will break with old models...
    model = instantiate_model(device, cfg)
    model.load_state_dict(mdict['model_state_dict'], strict=True)
    model = model.to(device)
    if eval_:
        model.eval()
    return model


## ---------- training helper fns ---------- ##

def inception_one_step(out, spec_true, loss, losstype):
    # get auxiliary loss too
    l1, aux = out
    loss_1 = calculate_loss(l1, spec_true, loss, losstype)
    loss_2 = calculate_loss(aux, spec_true, loss, losstype)
    total_loss = loss_1 + loss_2            
    return total_loss, loss_1, loss_2
    
def fivecrop_one_step(inputs, model):
    # if using the fivecrop augmentation, special case, average score across crops for image
    # taken from https://pytorch.org/vision/main/generated/torchvision.transforms.TenCrop.html
    imgs = TF.five_crop(inputs, size=(dataset.FC_SIZE,dataset.FC_SIZE))
    imgs = torch.stack(imgs)
    ncrops, bs, c, h, w = imgs.size()
    # fuse batch size and ncrops
    imgs = imgs.view(-1, c, h, w)
    specs = model(imgs)
    # now avg over crops
    specs = specs.view(bs, ncrops, -1).mean(dim=1)
    return specs.cpu()
    
# handles annoying need to convert targ to float
# type for loss calculation for BCE flavor losses
def calculate_loss(out, targ, loss, losstype):
    # checking loss type b/c BCE wants floats
    if losses[losstype] in [losses.BCE, losses.WEIGHTED_BCE, losses.CE, losses.WEIGHTED_CE]:
        return loss(out, targ.float())
    else:
        return loss(out, targ)

    
def train_one_epoch(model, train_loader, optimizer, loss, args, device, steps, tbwriter=None):

    
    for batch in tqdm(train_loader, total=len(train_loader), unit='batch'):
        # reset optimizer for next batch
        optimizer.zero_grad()
        # get contents
        spec_true, gen_true, fam_true, inputs = batch_to_gpu(batch, device, args)
        # setup spoofed loss outputs
        loss_gen, loss_fam = torch.tensor(0), torch.tensor(0)
        # get outputs
        out = model(inputs)
        # handle special cases for loss
        # special case for inception since it only uses the species information
        if mods[args.model] is mods.INCEPTION:
            total_loss, loss_1, loss_2 = inception_one_step(out, spec_true, loss, args.loss)
            if tbwriter is not None:
                tb_writer.add_scalar("train/final_loss", loss_1.item(), steps)
                tb_writer.add_scalar("train/aux_loss", loss_2.item(), steps)
                tb_writer.add_scalar("train/tot_loss", total_loss.item(), steps)
        else:
            # hacky, but calculates loss for each taxon
            if 'spec' in args.taxon_type:
                loss_spec = calculate_loss(out[0], spec_true, loss, args.loss)
            if 'gen' in args.taxon_type:
                loss_gen = calculate_loss(out[1], gen_true, loss, args.loss)
            if 'fam' in args.taxon_type:
                loss_fam = calculate_loss(out[2], fam_true, loss, args.loss)
            total_loss = loss_spec + loss_gen + loss_fam
            # report to tensorboard
            if tbwriter is not None:
                #if ALS loss, save the delta p for tuning focal hyperparameters
                if (losses[args.loss] is losses.ASL) or (losses[args.loss] is losses.SCALED_ASL):
                    record_deltaP(spec_true,out[0], tbwriter, steps, 'train', 'species')
                    record_deltaP(gen_true,out[1], tbwriter, steps, 'train', 'genus')
                    record_deltaP(fam_true,out[2], tbwriter, steps, 'train', 'family')
                tbwriter.add_scalar("train/spec_loss", loss_spec.item(), steps)
                tbwriter.add_scalar("train/gen_loss", loss_gen.item(), steps)
                tbwriter.add_scalar("train/fam_loss", loss_fam.item(), steps)
                tbwriter.add_scalar("train/tot_loss", total_loss.item(), steps)
        # actual gradient descent
        total_loss.backward()
        optimizer.step()
        steps+=args.batchsize

    return steps, optimizer, model


## ---------- testing helper fns ---------- ##

def logit_to_proba(y_pred, losstype):
    # softmax vs. sigmoid transformation
    if losses[losstype] is losses.CE or losses[losstype] is losses.WEIGHTED_CE:
        y_pred = torch.softmax(y_pred,axis=1)
    else:
        y_pred = torch.sigmoid(y_pred)
    y_obs = y_pred >= 0.5
    y_obs = y_obs.numpy()
    y_pred = y_pred.numpy()
    return y_obs, y_pred

def per_species_metrics(y_pred,y_true_multi):
    assert len(y_pred.shape) == 2, 'too many dimensions in probabilty vector!'
    # loop through each species, bit hacky
    j = 0
    aucs, prcs = [], []
    for i in range(y_pred.shape[1]):
        # this handles species not present in the test set (all absences)
        # https://github.com/scikit-learn/scikit-learn/pull/19085
        try:
            j += 1
            aucs.append(mets.roc_auc_score(y_true_multi[:,i], y_pred[:,i]))
            prcs.append(mets.average_precision_score(y_true_multi[:,i], y_pred[:,i]))
        except:
            aucs.append(np.nan)
            prcs.append(np.nan)
    aucmean = np.ma.MaskedArray(aucs, np.isnan(aucs)).mean()
    prcmean = np.ma.MaskedArray(prcs, np.isnan(prcs)).mean()
    return aucmean, prcmean
    
def test_model(model, test_loader, loss, args, device, test_steps, tbwriter=None):
    model.eval()
    # not calculating any gradients so can turn
    # off for any tensors generated inside
    with torch.no_grad():
        y_pred = []
        for batch in tqdm(test_loader, total=len(test_loader), unit='batch'):
            # set up batch, labels for loss calculation
            # only care about species-level prediction acc
            spec_true, _, _, inputs = batch_to_gpu(batch, device, args)
            # if using the fivecrop augmentation, special case, average score across crops for image
            specs = fivecrop_one_step(inputs, model) if args.augment == 'fivecrop' else model(inputs)
            specs = model(inputs)
            y_pred.append(specs)
            test_loss = calculate_loss(specs, spec_true, loss, args.loss)
            if tbwriter is not None:
                tbwriter.add_scalar("test/spec_loss", test_loss.item(), test_steps)
            test_steps += args.batchsize
    return y_pred, test_steps


def test_one_epoch(model, test_loader, test_dset, loss, shared_species, args, device, epoch, test_steps, tbwriter=None):
    # test (only if not using entire dataset for training)
    # be sure to turn off batchnorm et al for testing
    y_pred, test_steps = test_model(model, test_loader, loss, args, device, test_steps, tbwriter)
    # prepping for acc metrics
    y_pred = torch.cat(y_pred, dim=0)
    y_obs, y_pred = logit_to_proba(y_pred.cpu(), args.loss)
    y_pred_single = y_pred.copy()
    # filter to only present species
    # don't filter single species label so it matches w/ index
    y_pred_multi = y_pred[:,shared_species]
    y_obs = y_obs[:,shared_species]
    # get true multilabels
    # breaks if shuffle=True on dataloader!!
    y_true_multi = test_dset.all_specs_multi.numpy()
    y_true_single = test_dset.specs.numpy()
    # filter down to just the species
    # present in both datasets
    y_true_multi = y_true_multi[:,shared_species]
    # filter out rows from non-shared species
    mask = [True if sp in shared_species else False for sp in y_true_single]
    y_true_single = y_true_single[mask]
    y_pred_single =y_pred_single[mask,:]
    # ranking metrics
    lrap = label_ranking_average_precision_score(y_true_multi, y_pred_multi)
    top10, _ = utils.obs_topK(y_true_single, y_pred_single, 10)
    top30, _ = utils.obs_topK(y_true_single, y_pred_single, 30)
    top100, _ = utils.obs_topK(y_true_single, y_pred_single, 100)
    # class-based metrics
    aucmean, prcmean = per_species_metrics(y_pred_multi,y_true_multi)
    if tbwriter is not None:
        tbwriter.add_scalar(f"test/label_ranking_average_precision", lrap, epoch)
        tbwriter.add_scalar(f"test/top10_accuracy", top10, epoch)
        tbwriter.add_scalar(f"test/top30_accuracy", top30, epoch)
        tbwriter.add_scalar(f"test/top100_accuracy", top100, epoch)
        tbwriter.add_scalar(f"test/mean_ROC_AUC", aucmean, epoch)
        tbwriter.add_scalar(f"test/mean_PRC_AUC", prcmean, epoch)
    # standard binary metrics
    for score in [precision_score, recall_score, f1_score]:
        averages = ['macro', 'micro', 'weighted']
        for avg in averages:
            sc = score(y_true_multi, y_obs, average=avg, zero_division=0.0)
            if tbwriter is not None:
                tbwriter.add_scalar(f"test/{score.__name__}_{avg}", sc, epoch)
    return test_steps
        

## ---------- driver fns ---------- ##        

def train_model(args, rng):

    start = time.time()
    # setup dataset
    print("using all points for training")
    train_dset, shared_species = instantiate_datasets(args)
    
    #  only for debuggig
    if args.testing:
        # just take first 5K observations to speed up testing
        train_dset.len_dset = min([500, train_dset.len_dset])

    args = make_config(args, train_dset, shared_species)
    
    # set up device, model, losses
    device = f"cuda:{args.device}" if int(args.device) >= 0 else 'cpu'
    print(f"using device {device}")
    model = instantiate_model(device, args)
    loss = instantiate_loss(args, train_dset, device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # set up summary writer
    tb_writer = None if args.testing else SummaryWriter(comment=f"{args.exp_id}_allpoints")

    # re-set seed again since sometimes if different
    # architectures are used, the random number generator
    # will be on a different value for the shuffling
    if args.seed > 0:
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    # set up parallelized data loader here
    train_loader = DataLoader(train_dset, args.batchsize, shuffle=True, pin_memory=False, num_workers=args.processes, collate_fn=collate, drop_last=False)

    steps = 0
    for epoch in range(args.epochs):
        
        print(f"Starting epoch {epoch}")
        # have to turn on batchnorm and dropout again
        model.train()
        steps, optimizer, model = train_one_epoch(model, train_loader, optimizer, loss, args, device, steps, tbwriter=tb_writer)
        save_model(model, optimizer, epoch, args, steps)
        
    end = time.time()
    total = (end-start)/3600
    print(f"took {total} hours to run")
    if not args.testing:
        tb_writer.close()
        write_traintime(total, args, 'trainonly', device)
    

def train_and_test_model(args, rng):

    start = time.time()
    # setup dataset
    train_dset, test_dset, shared_species = instantiate_datasets(args)
    #  only for debugging
    if args.testing:
        # just take first 5K observations to speed up testing
        train_dset.len_dset = min([500, train_dset.len_dset])

    args = make_config(args, train_dset, shared_species)
    
    # set up device, model, losses
    device = f"cuda:{args.device}" if int(args.device) >= 0 else 'cpu'
    print(f"using device {device}")
    model = instantiate_model(device, args)
    loss = instantiate_loss(args, train_dset, device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # set up summary writer
    tb_writer = None if args.testing else SummaryWriter(comment=f"{args.exp_id}")

    # re-set seed again since sometimes if different
    # architectures are used, the random number generator
    # will be on a different value for the shuffling
    if args.seed > 0:
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)
    # set up parallelized data loader here
    train_loader = DataLoader(train_dset, args.batchsize, shuffle=True, pin_memory=False, num_workers=args.processes, collate_fn=collate, drop_last=False)
    test_loader = DataLoader(test_dset, args.batchsize, shuffle=False, pin_memory=False, num_workers=args.processes, collate_fn=collate, drop_last=False)
    
    test_steps, steps = 0, 0
    for epoch in range(args.epochs):
        
        print(f"Starting epoch {epoch}")
        # have to turn on batchnorm and dropout again
        model.train()
        steps, optimizer, model = train_one_epoch(model, train_loader, optimizer, loss, args, device, steps, tbwriter=tb_writer)
        save_model(model, optimizer, epoch, args, steps)
        test_steps = test_one_epoch(model, test_loader, test_dset, loss, shared_species, args, device, epoch, test_steps,  tbwriter=tb_writer)
        
    end = time.time()
    total = (end-start)/3600
    print(f"took {total} hours to run")
    if not args.testing:
        tb_writer.close()
        write_traintime(total, args, 'traintest', device)
    

## ---------- main module ---------- ##        

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    # required args
    args.add_argument('--dataset_name', type=str, required=True, help='Name of dataset file to use')
    args.add_argument('--dataset_type', type=str, required=True, help='Type of dataset to use (multispecies or single species)',choices=dataset.DatasetType.valid())
    args.add_argument('--taxon_type', type=str, required=True, help='Type of dataset to use (multispecies or single species)',choices=['spec_gen_fam', 'spec_gen', 'speconly'])
    args.add_argument('--datatype', type=str, required=True, help='What kind of data to train on', choices=dataset.DataType.valid())
    args.add_argument('--lr', type=float, required=True, help='what learning rate to use')        
    args.add_argument('--epochs', type=int, help='how many epochs to train the model for', required=True)
    args.add_argument('--model', type=str, required=True, help="what model to train", choices = mods.valid())
    args.add_argument('--exp_id', type=str, required=True, help="expermient id")
    args.add_argument('--loss', type=str, required=True, help='what loss function to use', choices=losses.valid())
    args.add_argument('--batchsize', type=int, required=True, help='batch size')
    # optional args
    args.add_argument('--year', type=str, help='What year of imagery to use as training data', default='2012')
    args.add_argument('--state', type=str, help='What state / region to train on', default='ca')
    args.add_argument('--band', type=int, default=-1, help='which band to use. -1 indicates that the spatial exclusion split of the data will be used')
    args.add_argument('--pretrain', type=str, help='What kind of pretraining to use', choices=tresnet.Pretrained.valid(), default='NONE')
    args.add_argument('--augment', type=str, help='What kind of data augmentation to use', choices=dataset.Augment.valid(), default='NONE')
    args.add_argument('--seed', type=int, help="Random seed for reproducibility", default=0)
    args.add_argument('--device', type=int, help="Which CUDA device to use. Set -1 for CPU", default=-1)
    args.add_argument('--processes', type=int, help="How many worker processes to use with the dataloader", default=1)
    # bool args
    args.add_argument('--testing', action='store_true', help="Whether to use a small subset of the data for training for debugging purposes")
    args.add_argument('--all_points', action='store_true', help="Whether to use all points in the dataset and not use any kind of split")

    args, _ = args.parse_known_args()
    # if seed is set, set all RNGs, and
    # turn off nondeterministic algorithms
    # inside of pytorch
    # https://pytorch.org/docs/stable/notes/randomness.html
    # for CUDA 10.2+
    # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    if args.seed > 0:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        rng = np.random.default_rng(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
        print("don't forget to set CUBLAS_WORKSPACE_CONFIG for CUDA 10.2+!")
        # To enable deterministic behavior in this case, you must set an environment variable before running your PyTorch application:
        # CUBLAS_WORKSPACE_CONFIG=:4096:8 or CUBLAS_WORKSPACE_CONFIG=:16:8.
        # For more information, go to https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
    else:
        rng = np.random.default_rng()
    if args.all_points:
        train_model(args, rng)
    else:
        train_and_test_model(args, rng)


