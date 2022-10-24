# deepbiosphere packages
import deepbiosphere.Run as run
import deepbiosphere.Utils as utils
import deepbiosphere.Models as mods
from deepbiosphere.Utils import paths
import deepbiosphere.Dataset as dataset
import deepbiosphere.NAIP_Utils as naip

# ML + statistics packages
import torch
import numpy as np
import pandas as pd
import sklearn.metrics as mets
from torch.utils.data import DataLoader

# miscellaneous packages
import os
import csv
import glob
import time
import json
import warnings
from tqdm import tqdm
from datetime import date
from os.path import exists
from tqdm import tqdm


def load_baseline_preds(nobs, nspecs, sp2id, model, band='unif_train_test', dset_name='big_cali_2012'):
    files = glob.glob(f"{paths.BASELINES}{model}/predictions/{dset_name}/{band}/*csv")
    results = np.zeros((nobs, nspecs))
    for file in tqdm(files):
        pred = pd.read_csv(file)
        if model == 'maxent':
            spec = file.split('/')[-1].split('_maxent_preds.csv')[0].replace('_', ' ')
            # fill in predictions to be in same order as CNN model
            results[:,sp2id[spec]] = pred.pres_pred
        elif model == 'rf':
            spec = file.split('/')[-1].split('_rf_preds.csv')[0].replace('_', ' ')
            results[:,sp2id[spec]] = pred.presence
        else:
            # TODO: figure out biomod probs
            raise NotImplemented
    return results

def write_overall_metric(dict_, sc, scorename, thres, weight):
    dict_['value'] = sc
    dict_['metric'] = scorename
    dict_['weight'] = weight
    dict_['thres'] = thres
    return dict_

def write_topk_metric(dict_, single_ytrue, preds, K, topKmet, type_):
    dict_['metric'] = f"{type_}_top{K}"
    dict_['weight'] = np.nan
    dict_['value'] =  topKmet(single_ytrue, preds, K)[0]
    return dict_

def write_spec_metric(dict_, metric, thres, vals, id2sp):
    dict_['metric'] = metric
    dict_['thres'] = thres
    re = {id2sp[i] : vals[i] for i in range(len(vals))}
    dict_.update(re)
    return dict_

def write_obs_metrics(dict_, metric, vals, ids, writer):
    dict_['metric'] = metric
    for v, id_ in zip(vals, ids):
        dict_['value'] = v.item()
        dict_['ID'] = id_
        writer.writerow(dict_)
def evaluate_model(ytrue, single_ytrue, preds, sharedspecs, sp2id, ids, dset_name, band, model, loss, lr, epoch, exp_id, pretrained, write_obs=False, thres=0.5, filename=None):
    tick = time.time()
    id2sp = {v:k for k, v in sp2id.items()}
    yobs = preds >= thres
    # make directory if it doesn't exist
    if not os.path.exists(f"{paths.RESULTS}accuracy_metrics/"):
        os.makedirs(f"{paths.RESULTS}accuracy_metrics/")
    # for locations with NaNs, impute
    # a probability of 0 at those locations
    # (only really relevant for baseline models)
    # because some sklearn functions will
    # handle the Nan
    if np.isnan(preds).sum() > 0:
        preds = np.ma.MaskedArray(preds, np.isnan(preds))        
    if np.ma.isMaskedArray(preds):
        preds = preds.filled(fill_value=0.0)
    fname = f"{paths.RESULTS}accuracy_metrics/overall_metrics_results_band{band}.csv" if filename is None else f"{paths.RESULTS}accuracy_metrics/{filename}overall_metrics_results_band{band}.csv"
    fexists = os.path.isfile(fname)
    overallcsv = open (fname, 'a')
    basics = {
        'value' : np.nan,
        'dset_name' : dset_name,
        'band' : band,
        'model' : model,
        'loss' : loss,
        'lr' : lr,
        'epoch' : epoch,
        'exp_id' : exp_id,
        'pretrained' : pretrained, 
        'metric' : np.nan,
        'weight' : np.nan,
        'thres' : thres,
        'date' : date.today(),
    }
    print("starting overall metrics")
    overallwriter = csv.DictWriter(overallcsv, delimiter=',', lineterminator='\n',fieldnames=basics.keys())
    if not fexists:
        overallwriter.writeheader()  # file doesn't exist yet, write a header

    # run + write overall binary accuracy metrics
    scores = [mets.precision_score, mets.recall_score, mets.f1_score, 
              mets.jaccard_score, mets.label_ranking_average_precision_score]
    for score in scores:
        if score == mets.label_ranking_average_precision_score:
            sc = score(ytrue, preds)
            overallwriter.writerow(write_overall_metric(basics, sc, score.__name__, np.nan, np.nan))
        else:
            averages = ['macro', 'micro', 'weighted', 'samples']
            for avg in averages:
                sc = score(ytrue[:,sharedspecs], yobs[:,sharedspecs], average=avg, zero_division=0.0)
                overallwriter.writerow(write_overall_metric(basics, sc, score.__name__, thres, avg))
    # run + write topK metrics 
    for i in [5,30,100]:
        overallwriter.writerow(write_topk_metric(basics, single_ytrue, preds, i, utils.obs_topK, 'obs'))
        overallwriter.writerow(write_topk_metric(basics, single_ytrue, preds, i, utils.species_topK, 'species'))
        # now, write out per-species metrics
    print("starting per-species metrics")
    fname = f"{paths.RESULTS}accuracy_metrics/per_species_metrics_results_band{band}.csv" if filename is None else f"{paths.RESULTS}accuracy_metrics/{filename}per_species_metrics_results_band{band}.csv"
    fexists = os.path.isfile(fname)
    csvfile = open (fname, 'a')
    dict_ = { k: np.nan for k,v in sp2id.items()}
    dict_.update(basics)
    # don't use these columns for species dict
    del dict_['weight'], dict_['value']
    writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=dict_.keys())
    if not fexists:
        writer.writeheader()  # file doesn't exist yet, write a header

    # run + write out roc-auc, prc-auc
    aucs = []
    prcs = []
    assert len(preds.shape) == 2, 'too many dimensions in probabilty vector!'
    j = 0
    for i in range(preds.shape[1]):
        # this handles species not present in the test set (all absences)
        # https://github.com/scikit-learn/scikit-learn/pull/19085
        try:
            j += 1
            aucs.append(mets.roc_auc_score(ytrue[:,i], preds[:,i]))
            prcs.append(mets.average_precision_score(ytrue[:,i], preds[:,i]))
        except:
            aucs.append(np.nan)
            prcs.append(np.nan)
    # also write out average AUCs
    aucmean = np.ma.MaskedArray(aucs, np.isnan(aucs)).mean()
    overallwriter.writerow(write_overall_metric(basics, aucmean, 'ROC_AUC', np.nan, np.nan))
    prcmean = np.ma.MaskedArray(prcs, np.isnan(prcs)).mean()
    overallwriter.writerow(write_overall_metric(basics, prcmean, 'PRC_AUC', np.nan, np.nan))
    # and calibrated AUCs
    cal_rocs, cal_prcs = utils.mean_calibrated_roc_auc_prc_auc(ytrue, yobs)
    cal_rocmean = np.ma.MaskedArray(cal_rocs, np.isnan(cal_rocs)).mean()
    overallwriter.writerow(write_overall_metric(basics, cal_rocmean, 'calibrated_ROC_AUC', np.nan, np.nan))
    cal_prcmean = np.ma.MaskedArray(cal_prcs, np.isnan(cal_prcs)).mean()
    overallwriter.writerow(write_overall_metric(basics, cal_prcmean, 'calibrated_PRC_AUC', np.nan, np.nan))
    overallcsv.close()
    
    # get individual species for topK spec
    for i in [5,30,100]:
        _, specs = utils.species_topK(single_ytrue, preds, i)
        writer.writerow(write_spec_metric(dict_, f'species_top{i}', i, specs, id2sp))
    
    a = mets.precision_recall_fscore_support(ytrue, yobs, zero_division=0)
    writer.writerow(write_spec_metric(dict_, 'ROC_AUC', np.nan, aucs, id2sp))
    writer.writerow(write_spec_metric(dict_, 'PRC_AUC', np.nan, prcs, id2sp))
    writer.writerow(write_spec_metric(dict_, 'calibrated_ROC_AUC', np.nan, cal_rocs, id2sp))
    writer.writerow(write_spec_metric(dict_, 'calibrated_PRC_AUC', np.nan, cal_prcs, id2sp))
    writer.writerow(write_spec_metric(dict_, 'precision_score', thres, a[0], id2sp))
    writer.writerow(write_spec_metric(dict_, 'recall_score', thres, a[1], id2sp))
    writer.writerow(write_spec_metric(dict_, 'f1_score', thres, a[2], id2sp))
    writer.writerow(write_spec_metric(dict_, 'support', thres, a[3], id2sp))
    csvfile.close()
    if write_obs:
        print('starting per-observation metrics')
        fname = f"{paths.RESULTS}accuracy_metrics/per_observations_metrics_results_band{band}.csv" if filename is None else f"{paths.RESULTS}accuracy_metrics/{filename}per_observations_metrics_results_band{band}.csv"
        fexists = os.path.isfile(fname)
        csvfile = open (fname, 'a')
        del basics['weight']
        basics['value'] = np.nan
        basics['ID'] = np.nan
        basics['thres'] = thres
        writer = csv.DictWriter(csvfile, delimiter=',', lineterminator='\n',fieldnames=basics.keys())
        if not fexists:
            writer.writeheader()  # file doesn't exist yet, write a header
        # finally, per-observation metrics
        basics['metric'] = f"precision_score"
        val = utils.precision_per_obs(yobs[:,sharedspecs], ytrue[:,sharedspecs])
        write_obs_metrics(basics, 'precision_score', val, ids, writer)
        val =  utils.recall_per_obs(yobs[:,sharedspecs], ytrue[:,sharedspecs])
        write_obs_metrics(basics, 'recall_score', val, ids, writer)
        val =  utils.accuracy_per_obs(yobs[:,sharedspecs], ytrue[:,sharedspecs])
        write_obs_metrics(basics, 'accuracy_perobs', val, ids, writer)
        val =  utils.f1_per_obs(yobs[:,sharedspecs], ytrue[:,sharedspecs])
        write_obs_metrics(basics, 'f1_score', val, ids, writer)
        
        # get individual species for topK obs
        # ignoring for now
        # for i in [5,30,100]:
        #     _, obs = utils.obs_topK(single_ytrue, preds, i)
        # itn's not zipping obs properly, maybe a shape problem TODO:
        #     write_obs_metrics(basics, f'obs_top{i}', obs, ids, writer)

        csvfile.close()  
    tock = time.time()
    return (tock - tick)/60

def run_inference(model, cfg, dloader, device, softmax_=False):

    y_pred = []
    for batch in tqdm(dloader, unit='batch'):
        # can ignore label
        _, _, _, inputs = batch
        # to handle joint model
        if cfg.datatype == 'joint_naip_bioclim':
            inputs = (inputs[0].float().to(device), inputs[1].float().to(device))
        else:
            inputs = inputs.float().to(device)
        # handle augmenting (annoying)
        if cfg.augment == 'fivecrop':
            # taken from https://pytorch.org/vision/main/generated/torchvision.transforms.TenCrop.html
            imgs = TF.five_crop(inputs, size=(dataset.FC_SIZE,dataset.FC_SIZE))
            imgs = torch.stack(imgs)
            ncrops, bs, c, h, w = imgs.size()
            # fuse batch size and ncrops
            imgs = imgs.view(-1, c, h, w)
            (specs, gens, fams) = model(imgs)
            # now avg over crops
            specs = specs.view(bs, ncrops, -1).mean(dim=1)
#             gens = gens.view(bs, ncrops, -1).mean(dim=1)
#             fams = fams.view(bs, ncrops, -1).mean(dim=1)
            y_pred.append(specs.cpu())
        elif (cfg.model == 'inception') or ('speconly' in cfg.model):
            out = model(inputs)
            y_pred.append(out.detach().cpu())
        # elif 'speconly' in cfg.model:
        #     y_pred.append(out.cpu())
        else:
            spec, _, _ = model(inputs)
            y_pred.append(spec.detach().cpu())
    y_pred = torch.cat(y_pred, dim=0)
    if softmax_:
        if cfg.loss in ['CPO', 'CE', 'CEWeighted']:
            y_pred = torch.softmax(y_pred, dim=1)
        else:
            y_pred = torch.sigmoid(y_pred)
    else:
        y_pred = torch.sigmoid(y_pred)
    return y_pred.numpy()
