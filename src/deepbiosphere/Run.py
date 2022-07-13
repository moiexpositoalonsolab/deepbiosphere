# GIS packages
import time
from csv import DictWriter
from datetime import date
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, LineString, box
from torch.utils.data import Dataset
# deepbiosphere packages
import deepbiosphere.Models as mods
import deepbiosphere.Utils as utils
import deepbiosphere.Losses as losses
import deepbiosphere.Dataset as dataset
from deepbiosphere.Utils import paths
import torch.nn as nn
import torchvision.transforms.functional as TF
import argparse
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score, f1_score, label_ranking_average_precision_score
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# statistics packages
import torch
import numpy as np
import scipy.stats
import pandas as pd
import torch.optim as optim
# trying to make random number gernerator more reproducible
# https://stackoverflow.com/questions/62309424/does-numpy-random-seed-always-give-the-same-random-number-every-time

from numpy.random import default_rng
from torch.utils.tensorboard import SummaryWriter
import random
# miscellaneous packages
import os
import copy
import tqdm
import time
import glob
import math
import glob
import json
import shutil
from tqdm import tqdm
from os.path import exists
from types import SimpleNamespace

 # Global dictionaries with the
# models to compare
MODELS = {
    'tresnet_m' : mods.TResNet_M,
    'tresnet_m_speconly' : mods.TResNet_M,
    'joint_tresnet_m' : mods.Joint_TResNet_M,
    'tresnet_l' : mods.TResNet_L,
    'joint_tresnet_l' : mods.Joint_TResNet_L,
    'mlp' : mods.Bioclim_MLP,
    'inception' : mods.InceptionV3
     }

# memory inefficient, means these losses are being instantiated every time
# Run is loaded, TODO fix!!
LOSSES = {
    'ASL' : losses.AsymmetricLoss(),
    'ASLScaled' : losses.AsymmetricLossScaled(),
    "BCE" : nn.BCEWithLogitsLoss(),
    "BCEScaled" : losses.AsymmetricLossScaled(gamma_neg=0, gamma_pos=0, clip=0.0),
    'BCEProbScaled' : losses.AsymmetricLossScaled(gamma_neg=1, gamma_pos=1, clip=0.0),
    'FocalLossScaled' : losses.AsymmetricLossScaled(gamma_neg=4, gamma_pos=1, clip=0.0),
    'BCEProbClipScaled' : losses.AsymmetricLossScaled(gamma_neg=1, gamma_pos=1, clip=0.05),
    "BCEWeighted" : losses.BCEWeighted,
    'CPO' : losses.CrossEntropyPresenceOnly(),
    'CE' : nn.CrossEntropyLoss(),
    'CEWeighted' : losses.CEWeighted,
     }

DATASETS = {
    'naip' : dataset.NAIP,
    'bioclim': dataset.Bioclim,
    'joint_naip_bioclim' : dataset.BioclimNAIP
}

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

    
def load_model(device, model, pret, loss, exp_id, lr, nspec, ngen, nfam, epoch, eval_=True):
    model_path= f"{paths.MODELS}{model}_{loss}/{exp_id}_lr{str(lr).split('.')[-1]}_e{epoch}.tar"
    mdict =  torch.load(model_path)
    model = MODELS[model](pret, num_spec=nspec, num_gen=ngen, num_fam=nfam, base_dir=paths.MODELS)
    model.load_state_dict(mdict['model_state_dict'], strict=True)
    model = model.to(device)
    if eval_:
        model.eval()
    return model   

def load_config(exp_id, band, loss, model):
    path = f"{paths.MODELS}configs/{model}_{loss}_band{band}_{exp_id}.json"
    with open(path, 'r') as f:
        return SimpleNamespace(**json.load(f))
    
def run(args, rng):

    start = time.time()
    train_dset = DATASETS[args.datatype](args.dataset_name, args.dataset_type, args.state, args.year, args.band, 'train', args.latname, args.loname, args.idCol, args.augment)
    test_dset = DATASETS[args.datatype](args.dataset_name, args.dataset_type, args.state, args.year, args.band, 'test', args.latname, args.loname, args.idCol, 'none')

    #  only for debuggig
    if args.testing:
        # just take first 5K observations to speed up testing
        train_dset.dataset = train_dset.dataset[:5000]


    # figure out what species are present in both the train and the test split
    # and only calculate accuracy metrics for those shared species
    train_specs = [spec for sublist in train_dset.dataset.specs_overlap_id for spec in sublist]
    test_specs = [spec for sublist in test_dset.dataset.specs_overlap_id for spec in sublist]
    shared_species = list(set(train_specs) & set(test_specs))
    # sanity check that there's no observation leakage
    for index in train_dset.index:
        # check for data leakage
        assert index not in test_dset.index; 'index overlap!'
    # another sanity check, make sure n_specs are identical
    # only need to check species bc genus and family will match
    assert train_dset.nspec == test_dset.nspec, "number of species don't match!"

    # save arguments
    path = f"{paths.MODELS}configs/{args.model}_{args.loss}_band{args.band}_{args.exp_id}.json"
    # if this is the first time building the config directory
    # make the directories and save out to disk
    currdir = os.path.dirname(path)
    if not os.path.exists(currdir):
        os.makedirs(currdir)
    with open(path, 'w') as f:
        json.dump(vars(args), f, indent=4)

    # set up device, model, losses
    device = f"cuda:{args.device}" if int(args.device) >= 0 else 'cpu'
    print(f"using device {device}")
    # MODELS is a global dictionary in the CNNwith keys as
    # simple names and values as the class type, so we can just call the model
    # name as a function
    if 'speconly' in args.model:
        model =  MODELS[args.model](args.pretrain, num_spec=train_dset.nspec, num_gen=-1, num_fam=-1, base_dir=paths.MODELS)
    elif 'joint' in args.model:
        model = MODELS[args.model](args.pretrain, num_spec=train_dset.nspec, num_gen=train_dset.ngen, num_fam=train_dset.nfam, env_rasters=train_dset.nrasters, base_dir=paths.MODELS) 
    else:
        model = MODELS[args.model](args.pretrain, num_spec=train_dset.nspec, num_gen=train_dset.ngen, num_fam=train_dset.nfam, base_dir=paths.MODELS) 
    model = model.to(device)
    spec_loss= LOSSES[args.loss] if "Weighted" not in args.loss else LOSSES[args.loss](train_dset.metadata['species_counts'],train_dset.metadata['spec_2_id'], train_dset.total_len, device)
    gen_loss = LOSSES[args.loss]  if "Weighted" not in args.loss else LOSSES[args.loss](train_dset.metadata['genus_counts'],train_dset.metadata['gen_2_id'], train_dset.total_len, device)
    fam_loss = LOSSES[args.loss]  if "Weighted" not in args.loss else LOSSES[args.loss](train_dset.metadata['family_counts'],train_dset.metadata['fam_2_id'], train_dset.total_len, device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if not args.testing:
        tb_writer = SummaryWriter(comment=f"{args.exp_id}")

# re-set seed again since sometimes if different
# architectures are used, the random number generator
# will be on a different value for the shuffling
    if args.seed > 0:
        torch.cuda.manual_seed(args.seed)
        torch.manual_seed(args.seed)

    # set up parallelized data loader here
    train_loader = DataLoader(train_dset, args.batchsize, shuffle=True, pin_memory=False, num_workers=args.processes, collate_fn=collate, drop_last=False)
    test_loader = DataLoader(test_dset, args.batchsize, shuffle=False, pin_memory=False, num_workers=args.processes, collate_fn=collate, drop_last=False)

    steps = 0
    test_steps = 0
    for epoch in range(args.epochs):
        # have to turn on batchnorm and dropout again
        model.train()
        for b in tqdm(train_loader, total=len(train_loader), unit='batch'):
            # get contents
            spec_true, gen_true, fam_true, inputs = b
            spec_true = spec_true.to(device)
            gen_true = gen_true.to(device)
            fam_true = fam_true.to(device)
            # to handle joint model
            if torch.is_tensor(inputs):
                inputs = inputs.float().to(device)
            else:
                inputs = (inputs[0].float().to(device), inputs[1].float().to(device))
            optimizer.zero_grad()
            out = model(inputs)
            # BCE requires float for targets for some reason
            if 'BCE' in args.loss:
                (specs, gens, fams) = out
                loss_spec = spec_loss(specs, spec_true.float())
                loss_gen = gen_loss(gens, gen_true.float())
                loss_fam = fam_loss(fams, fam_true.float())
                total_loss = loss_spec + loss_gen + loss_fam
            # edge case for inception since it only uses the species
            elif args.model =='inception':
                l1, aux = out
                loss_1 = spec_loss(l1, spec_true)
                loss_2 = gen_loss(aux, spec_true)
                total_loss = loss_1 + loss_2
                # spoof losses for tbwriter
                loss_spec = loss_1
                loss_gen = loss_2
                loss_fam = torch.tensor(0)
            # again, only species model
            elif 'speconly' in args.model:
                spec = out
                loss = spec_loss(spec, spec_true)
                total_loss = loss
                # spoof for tensorboard
                loss_gen = torch.tensor(0)
                loss_fam = torch.tensor(0)
            else:
                (specs, gens, fams) = out
                loss_spec = spec_loss(specs, spec_true)
                loss_gen = gen_loss(gens, gen_true)
                loss_fam = fam_loss(fams, fam_true)
                total_loss = loss_spec + loss_gen + loss_fam
            total_loss.backward()
            optimizer.step()
            if not args.testing:
                #if ALS loss, save the delta p for tuning focal hyperparameters
                if 'ASL' in args.loss:
                    record_deltaP(spec_true,specs, tb_writer, steps, 'train', 'species')
                    record_deltaP(gen_true,gens, tb_writer, steps, 'train', 'genus')
                    record_deltaP(fam_true,fams, tb_writer, steps, 'train', 'family')
                tb_writer.add_scalar("train/spec_loss", loss_spec.item(), steps)
                tb_writer.add_scalar("train/gen_loss", loss_gen.item(), steps)
                tb_writer.add_scalar("train/fam_loss", loss_fam.item(), steps)
                tb_writer.add_scalar("train/tot_loss", total_loss.item(), steps)
            steps+=args.batchsize
        # test
        # be sure to turn off batchnorm et al for testing
        model.eval()
        # not calculating any gradients so can turn
        # off for any tensors generated inside
        with torch.no_grad():
            y_pred = []
            for b in tqdm(test_loader, total=len(test_loader), unit='batch'):
                # set up batch, labels for loss calculation
                spec_true, gen_true, fam_true, inputs = b
                spec_true = spec_true.to(device)
                gen_true = gen_true.to(device)
                fam_true = fam_true.to(device)
                # to handle joint model
                if torch.is_tensor(inputs):
                    inputs = inputs.float().to(device)
                else:
                    inputs = (inputs[0].float().to(device), inputs[1].float().to(device))
                # if using the fivecrop augmentation, special case, average score across crops for image
                if args.augment == 'fivecrop':
                    # taken from https://pytorch.org/vision/main/generated/torchvision.transforms.TenCrop.html
                    imgs = TF.five_crop(inputs, size=(dataset.FC_SIZE,dataset.FC_SIZE))
                    imgs = torch.stack(imgs)
                    ncrops, bs, c, h, w = imgs.size()
                    # fuse batch size and ncrops
                    imgs = imgs.view(-1, c, h, w)
                    (specs, gens, fams) = model(imgs)
                    # now avg over crops
                    specs = specs.view(bs, ncrops, -1).mean(dim=1)
                    gens = gens.view(bs, ncrops, -1).mean(dim=1)
                    fams = fams.view(bs, ncrops, -1).mean(dim=1)
                    y_pred.append(specs.cpu())
                else:
                    out = model(inputs)
                # get loss on test set as well
                # BCE requires float for targets for some reason
                if 'BCE' in args.loss:
                    (specs, gens, fams) = out
                    y_pred.append(specs.cpu())
                    loss_spec = spec_loss(specs, spec_true.float())
                    loss_gen = gen_loss(gens, gen_true.float())
                    loss_fam = fam_loss(fams, fam_true.float())
                    total_loss = loss_spec + loss_gen + loss_fam
                elif args.model == 'inception':
                    y_pred.append(out.cpu())
                    total_loss = spec_loss(out, spec_true)
                    loss_spec = total_loss
                    # spoof other two losses
                    loss_gen = torch.tensor(0)
                    loss_fam = torch.tensor(0)
                elif 'speconly' in args.model:
                    y_pred.append(out.cpu())
                    loss_spec = spec_loss(out, spec_true)
                    total_loss = loss_spec
                    loss_gen = torch.tensor(0)
                    loss_fam = torch.tensor(0)
                else:
                    (specs, gens, fams) = out
                    y_pred.append(specs.cpu())
                    loss_spec = spec_loss(specs, spec_true)
                    loss_gen = gen_loss(gens, gen_true)
                    loss_fam = fam_loss(fams, fam_true)
                    total_loss = loss_spec + loss_gen + loss_fam
                if not args.testing:
                    #if ALS loss, save the delta p for tuning focal hyperparameters
                    if 'ASL' in args.loss:
                        record_deltaP(spec_true,specs, tb_writer, test_steps, 'test', 'species')
                        record_deltaP(gen_true,gens, tb_writer, test_steps, 'test', 'genus')
                        record_deltaP(fam_true,fams, tb_writer, test_steps, 'test', 'family')
                    tb_writer.add_scalar("test/spec_loss", loss_spec.item(), test_steps)
                    tb_writer.add_scalar("test/gen_loss", loss_gen.item(), test_steps)
                    tb_writer.add_scalar("test/fam_loss", loss_fam.item(), test_steps)
                    tb_writer.add_scalar("test/tot_loss", total_loss.item(), test_steps)
                test_steps += args.batchsize
            # prepping for sklearn metrics
            y_pred = torch.cat(y_pred, dim=0)
            y_pred = torch.sigmoid(y_pred)
            y_obs = y_pred >= 0.5
            y_pred = y_pred.numpy()
            y_obs = y_obs.numpy()
            y_true = test_dset.all_specs.numpy()
            # filter down to just the species
            # present in both datasets
            y_obs = y_obs[:,shared_species]
            y_true = y_true[:,shared_species]
            # ranking metrics
            sc = label_ranking_average_precision_score(test_dset.all_specs, y_pred)
            top10 = utils.obs_topK(test_dset.specs, y_pred, 10)
            top30 = utils.obs_topK(test_dset.specs, y_pred, 30)
            if not args.testing:
                tb_writer.add_scalar(f"test/label_ranking_average_precision", sc, epoch)
                tb_writer.add_scalar(f"test/top10_accuracy", top10, epoch)
                tb_writer.add_scalar(f"test/top30_accuracy", top30, epoch)
            # standard binary metrics
            scores = [precision_score, recall_score, f1_score]
            for score in scores:
                averages = ['macro', 'micro', 'weighted']
                for avg in averages:
                    sc = score(y_true, y_obs, average=avg, zero_division=0.0)
                    if not args.testing:
                        tb_writer.add_scalar(f"test/{score.__name__}_{avg}", sc, epoch)
        # finally, save out the model
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
    tb_writer.close() if not args.testing else None
    end = time.time()
    total = (end-start)/3600
    print(f"took {total} hours to run")
    towrite = {
        'time (in hours)' : total,
        'date' : date.today(),
        'code' : 'python',
        'model' : args.model,
        'dataset' : args.dataset_name,
        'band' : args.band,
        'gpu_mem' : torch.cuda.memory_allocated(device)
    }
    # save how long it took to run
    headers = ['time (in hours)', 'date', 'code', 'model', 'dataset', 'band', 'gpu_mem' ]
    fname = f"{paths.MODELS}timing_runs.csv"
    fexists = os.path.isfile(fname)
    with open(fname, 'a', newline='\n') as f:
        dctw = DictWriter(f, fieldnames=headers)
        # Pass the data in the dictionary as an argument into the writerow() function
        if not fexists:
            dctw.writeheader() # write header if not yet written
        dctw.writerow(towrite)



if __name__ == "__main__":

    args = argparse.ArgumentParser()

    args.add_argument('--year', type=str, help='What year of imagery to use as training data', default='2012')
    args.add_argument('--state', type=str, help='What state / region to train on', default='ca')
    args.add_argument('--dataset_name', type=str, required=True, help='Name of dataset file to use')
    args.add_argument('--dataset_type', type=str, required=True, help='Type of dataset to use (multispecies or single species)',choices=['multi_species', 'single_species', 'single_label'])
    args.add_argument('--datatype', type=str, required=True, help='What kind of data to train on', choices=DATASETS.keys()) 
    args.add_argument('--pretrain', type=str, required=True, help='What kind of pretraining to sue', choices=['none', 'imagenet', 'mscoco']) 
    args.add_argument('--band', type=int, default=-1, help='which band to use. -1 indicates that the spatial exclusion split of the data will be used')
    args.add_argument('--lr', type=float, required=True, help='what learning rate to use')
    args.add_argument('--latname', type=str, help='Name of the column that contains latitude information', default='decimalLatitude')
    args.add_argument('--testing', action='store_true', help="Whether to use a small subset of the data for training for debugging purposes")
    args.add_argument('--loname', type=str, help='Name of the column that contains latitude information', default='decimalLongitude')
    args.add_argument('--augment', type=str, help='What kind of data augmentation to use', choices=['none', 'random', 'fivecrop'])
    args.add_argument('--idCol', type=str, help='Name of the column thats used as the key in the image dictionary', default='gbifID')
    args.add_argument('--epochs', type=int, help='how many epochs to train the model for')
    args.add_argument('--model', type=str, required=True, help="what model to train", choices=MODELS.keys())
    args.add_argument('--exp_id', type=str, required=True, help="expermient id")
    args.add_argument('--loss', type=str, required=True, help='what loss function to use', choices=LOSSES.keys())
    args.add_argument('--batchsize', type=int, required=True, help='batch size')
    args.add_argument('--seed', type=int, help="Random seed for reproducibility", default=0)
    args.add_argument('--device', type=int, help="Which CUDA device to use. Set -1 for CPU", default=-1)
    args.add_argument('--processes', type=int, help="How many worker processes to use with the dataloader")
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
    run(args, rng)


