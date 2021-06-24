import os
import json
import time
import gc
import torch
import shutil
import pandas as pd
import numpy as np
from deepbiosphere.scripts import GEOCLEF_Utils as utils
from deepbiosphere.scripts import GEOCLEF_Dataset as dataset
from deepbiosphere.scripts import GEOCLEF_Config as config
from deepbiosphere.scripts.GEOCLEF_Config import paths
import deepbiosphere.scripts.GEOCLEF_Run as run
import sklearn.metrics as metrics


EXTRA_COLUMNS = [
    'NA_L1NAME',
     'NA_L2NAME',
     'NA_L3NAME',
     'US_L3NAME',
     'city',
     'lat',
     'lon',
     'region',
     'test'
    ]

# this assumes that all models we're running metrics on use the same dataset
# and the same number of top species (num_species)s
def run_metrics_and_longform(args):


    # get directory for all these runs
    save_dir = config.get_res_dir(args.base_dir)
    base_dir = args.base_dir
    # take a relative path to a file within GeoCELF2020 that a file that contains a dict of the configs
    # then, move this file into the results directory so that the names are saved with the resulting dataframes
    cfg_pth = args.base_dir + args.config_path
    filename = cfg_pth.split('/')[-1]
    shutil.copyfile(cfg_pth, save_dir + filename)    
    cfg_pth = save_dir + filename
    with open(cfg_pth, 'r') as fp:
        cfgs = json.load(fp)
    # load configs in
#     print(cfgs)
#     print("---")

    models = cfgs['models']
    params = load_configs(models, base_dir)
    g_t = load_configs({"ground_truth" : cfgs['ground_truth']}, base_dir)
    prm = list(params.values())[0]
    num_specs = args.num_species
    dset= run.setup_dataset(prm.params.observation, args.base_dir, prm.params.organism, prm.params.region, prm.params.normalize, prm.params.no_altitude, prm.params.dataset, prm.params.threshold, num_species=num_specs)
    # load data from configs
    all_df, ground_truth = load_data(params, g_t['ground_truth'], args.which_taxa)
 
    # add threshold parameter to file so it's saved for future use
    if 'pres_threshold' not in cfgs.keys():
        cfgs['pres_threshold'] = args.threshold
        with open(cfg_pth, 'w') as fp:
            json.dump(cfgs, fp)
            fp.close()
        threshold = args.pres_threshold
    else: 
        threshold = cfgs['pres_threshold']
    if 'ecoregion' not in  cfgs.keys():
        cfgs['ecoregion'] = args.ecoregion
        with open(cfg_pth, 'w') as fp:
            json.dump(cfgs, fp)
            fp.close()
        ecoregion = args.ecoregion
    else:
        ecoregion = cfgs['ecoregion']

    device = args.device
    # check what indices are train / test
    # check column labels are correct
    # get rid of extra columns laying around
    for name, dic in all_df.items():
        for mod, df in dic.items():
            if 'Unnamed: 0' in df.columns:
                            df.pop( 'Unnamed: 0')
    for taxa, df in ground_truth.items():
        if 'Unnamed: 0' in df.columns:
                df.pop( 'Unnamed: 0')
    # get labels of each unique taxa in data
#     print(all_df.keys(), "---")
    taxa_names = check_colum_names(dset, all_df)
    # convert logits and probabilities to presences and absences based on probabilities
    pres_df = pred_2_pres(all_df, taxa_names, device, threshold)

    # metrics to use
    overall_mets = [
    metrics.precision_score,
    metrics.recall_score,
    metrics.f1_score,
    metrics.accuracy_score,
    #metrics.roc_auc_score
    ]
    mets = [
    metrics.precision_score,
    metrics.recall_score,
    metrics.f1_score,
    metrics.accuracy_score
    ]

    # TODO: handle these bad boys
#     mets_extra = [
#     metrics.roc_curve, # proba    
#     metrics.confusion_matrix # pres-abs
#     ]
    
    # run all per-label metrics globally

    per_spec_glob_mets = sklearn_per_taxa_overall(pres_df, ground_truth, overall_mets, taxa_names)
    pth = save_dir + "per_species_overall_{}.csv".format(cfgs['exp_id'])
    print("savial:::dfasfsadsadg to:", pth)    
    per_spec_glob_mets.to_csv(pth)
    print("global metrics done")
    # run all per-label metrics within ecoregions
#     ecoregion = args.ecoregion
    per_spec_eco_mets = sklearn_per_taxa_ecoregion(pres_df, ground_truth, mets, taxa_names, ecoregion)
    pth = save_dir + "per_species_eco_{}_{}.csv".format(ecoregion, cfgs['exp_id'])
    per_spec_eco_mets.to_csv(pth)
    print("per-ecoregion metrics done")
#     run all per-label metrics and preserve for all labels
    for taxa in pres_df.keys():
        per_spec_all = sklearn_per_taxa_individual(pres_df[taxa], ground_truth[taxa], taxa_names[taxa])
        pth = save_dir + "per_{}_by_{}_{}.csv".format(taxa, taxa, cfgs['exp_id'])
        per_spec_all.to_csv(pth)
    print("per-species metrics done")
    # run all observation
    for taxa in pres_df.keys():
#         print(taxa_names[taxa])
        per_obs_all = inhouse_per_observation(pres_df[taxa], ground_truth[taxa], taxa_names[taxa], args.device)
        
        pth = save_dir + "per_obs_by_{}_{}.csv".format(taxa, cfgs['exp_id'])
        print("saving to ", pth)
        per_obs_all.to_csv(pth)
    print("per-observation metrics done")
        # TODO: add mets_extra
    
def load_configs(cfgs, base_dir):
    # get these configs
    params = {}
    for name, cfg in cfgs.items():
#         print(cfg)
        param = config.Run_Params(base_dir = base_dir, cfg_path = cfg)
        params[name] = param
    return params
    
def load_data(params, ground_truth, which_taxa):    
    # load these configs' inference data
    print("loading ground truth")
    spt, gent, famt = ground_truth.get_most_recent_inference()
    if which_taxa == 'spec_gen_fam':
        g_t = {
            'species' : pd.read_csv(spt),
            'genus' : pd.read_csv(gent),
            'family' : pd.read_csv(famt)
        }
    elif which_taxa =='spec_only':
        g_t = {
            'species' : pd.read_csv(spt)
            }
    else:
        raise NotImplementedError("not yet impletmented for ", which_taxa)
    data_s = {}
    data_g = {}
    data_f = {}
    ground_truth  ={}
    for name, param in params.items():
        print("loading model ", name)
        tick = time.time()
        if which_taxa == 'spec_only':
            sp = param.get_most_recent_inference(which_taxa=which_taxa)
            # do spec only
            data_s[name] = pd.read_csv(sp)
        elif which_taxa == 'spec_gen_fam':
            sp, gen, fam = param.get_most_recent_inference(which_taxa=which_taxa)
            # do spgenfam
            data_s[name] = pd.read_csv(sp)
            data_g[name] = pd.read_csv(gen)
            data_f[name] = pd.read_csv(fam)
        else:
            raise NotImplementedError('inference not yet implemented for ', which_taxa)
        tock = time.time()
        print("loading {} took {} minutes".format(name, ((tock-tick)/60)))
    if which_taxa == 'spec_only':
    
        all_df = {
        'species' : data_s,
    }
    elif which_taxa == 'spec_gen_fam':
        all_df = {
        'species' : data_s,
        'genus' : data_g,
        'family' : data_f
    }
    else:
        raise NotImplementedError('inference not yet implemented for ', which_taxa)
    return all_df, g_t
    

def check_colum_names(dset, all_df):
    
    species_columns, genus_columns, family_columns = None, None, None
    for name, taxa in all_df.items():
#         print("name is ", name)
#         print("----")
        col_labels = []        
        for n, model in taxa.items():
            col_labels.append(list(model.columns))
        to_comp = col_labels[0]
        for i in col_labels:
            assert to_comp == i, "column names are not consistent!"
        for i in col_labels:
            if name == 'species':
                specs = set(i) - set(EXTRA_COLUMNS)
#                 print(len(specs), dset.num_specs)
                assert len(specs) == dset.num_specs, "unexpected columns present in species inference file!"
                species_columns = specs
            elif name == 'genus':
                specs = set(i) - set(EXTRA_COLUMNS)
#                 print(len(specs), dset.num_gens)                      
                assert len(specs) == dset.num_gens, "unexpected columns present in genus inference file!"
                genus_columns = specs                
            else:
                specs = set(i) - set(EXTRA_COLUMNS)
#                 print(len(specs), dset.num_fams)
                assert len(specs) == dset.num_fams, "unexpected columns present in family inference file!"
                family_columns = specs                
    taxa_names = {
        'species' : species_columns,
        'genus' : genus_columns,
        'family' : family_columns
    }
    return taxa_names

# 1. convert predictions to presence / absence
def pred_2_pres(all_df, taxa_names, device, threshold):
    pres_df = {}
    for taxa, dic in all_df.items():
        new_dict = {}

        for name, df in dic.items():
            cols_2_keep = list(taxa_names[taxa])
            obs = df[cols_2_keep].values
            obs = torch.tensor(obs)
            obs.to(device)
            if name == 'random_forest' or name == 'maxent':
                # already a probability, just threshold
                binn = (obs > threshold).float()
            else:
                # first convert logit to probability
                obs = torch.sigmoid(obs)
                # then threshold probability
                # sneaky: NaN is a very small double, but when you convert it to float32 with .float()
                # then it gets rounded to a very negative number, but no longer gets mapped to NaN
                # therefore, have to make sure to convert the bool array to a double so that 
                # NaN statuts gets carried over and preserved
                binn = (obs > threshold).double()
                # sketchy, but the below line essentially converts over the nans if inference
                # only run on either test or  train data, so that metrics aren't accidentally
                # calculated for portions of the dataset inference wasn't actually run on
                binn[torch.isnan(obs)] = obs[torch.isnan(obs)]
            # convert back to numpy and df
            out = binn.cpu().numpy()
            # now this new df will have NaN values for any observations that inference wasn't run on 
            new_dict[name] = utils.numpy_2_df(out, taxa_names[taxa], df, EXTRA_COLUMNS)
        pres_df[taxa] = new_dict
    return pres_df


# 1. convert predictions to probabilities
def pred_2_proba(all_df, taxa_names, device):
    prob_df = {}
    for taxa, dic in all_df.items():
        new_dict = {}    
        for name, df in dic.items():
            if name != 'random_forest' and name != 'maxent':    
                col_2_keep = list(taxa_names[taxa])
                obs = df[col_2_keep].values
                obs = torch.tensor(obs)
                obs.to(device)
                obs = torch.sigmoid(obs)
            # convert back to numpy and df
                out = obs.cpu().numpy()
                new_dict[name] = utils.numpy_2_df(out, col_2_keep, df, EXTRA_COLUMNS)
            else:
                # random forest and maxent are already probabilities, just pass on
                new_dict[name] = df
            prob_df[taxa] = new_dict

    return prob_df    
# per-species datasaet-wide
# 2. sklearn statistics
# test
# to get per-class: micro/macro/weighted to get per-sample: samples
# model  test taxa met1, met2, ... metn
# maxent  T   spec 1.2  3.4        3.3 
# maxent  F   fam 1.2  3.4        3.3
# tresnet F   spec 1.2  3.4        3.3
# ...  


def sklearn_per_taxa_overall(pres_df, ground_truth, mets, taxa_names):
    num_models = len(list(pres_df.values())[0].keys())
    num_taxa = len(pres_df)
    num_rows = num_models * num_taxa *2 # the two is one for test, one for train
    score_name = [s.__name__ for s in mets]
    extra = ['model', 'taxa', 'test'] 
    score_idx = {score : (i + len(extra)) for score, i in zip(score_name, range(len(score_name)))}
    df_names = extra + score_name
    model_idx, taxa_idx, test_idx = 0, 1, 2
    filler = np.zeros([num_rows, len(df_names)])
    results_df = pd.DataFrame(filler, columns=df_names)
    # TODO: will break if there aren't observations for train points.../def
    i = 0
    # one entry per-taxa
    for taxa, dic in pres_df.items():
        # one entry per-model
        for name, df in dic.items():
            col_2_keep = list(taxa_names[taxa])
            gt = ground_truth[taxa]
            # grab test predictions
            curr_df =  df[df.test]
            obs = curr_df[col_2_keep].values
            curr_gt = gt[gt.test]
            curr_yt =  curr_gt[col_2_keep].values
            results_df = run_metrics(results_df, curr_yt, obs, mets, score_idx, i)
            results_df.iloc[i, taxa_idx] = taxa
            results_df.iloc[i, test_idx] = 'test'
            results_df.iloc[i, model_idx] = name
            i += 1
            # train
            curr_df =  df[~df.test]
            obs = curr_df[col_2_keep].values
            curr_gt = gt[~gt.test]
            curr_yt =  curr_gt[col_2_keep].values
            # presence / absence metrics
            results_df = run_metrics(results_df, curr_yt, obs, mets, score_idx, i)
            results_df.iloc[i, taxa_idx] = taxa
            results_df.iloc[i, test_idx] = 'train'
            results_df.iloc[i, model_idx] = name    
            i += 1
    return results_df


def run_metrics(df, y_true, y_obs, mets, score_idx, i):
# probability weight metrics
    for met in mets:
        tick = time.time()
        if met.__name__ == 'accuracy_score' or met.__name__ == 'roc_auc' :
            out = met(y_true, y_obs) 
        else:
            out = met(y_true, y_obs, average='weighted') 
        idx = score_idx[met.__name__]
        df.iloc[i, idx] = out
        tock = time.time()
        print("{} took {} minutes".format(met.__name__, ((tock-tick)/60)))
    return df
    
# per-taxa across egoregions
# 2. sklearn statistics
# test
# to get per-class: micro/macro/weighted to get per-sample: samples
# model eco test taxa met1, met2, ... metn
# maxent E1  T  spec 1.2  3.4        3.3 
# maxent E2  F   fam 1.2  3.4        3.3
# tresnet E1 F   spec 1.2  3.4        3.3
# ...  
# per-species datasaet-wide
# 2. sklearn statistics
# test
# to get per-class: micro/macro/weighted to get per-sample: samples
# model  test taxa met1, met2, ... metn
# maxent  T   spec 1.2  3.4        3.3 
# maxent  F   fam 1.2  3.4        3.3
# tresnet F   spec 1.2  3.4        3.3
# ...  

def sklearn_per_taxa_ecoregion(pres_df, ground_truth, mets, taxa_names, econame):
    
    # setting up dataframe
    num_models = len(list(pres_df.values())[0].keys())
    num_taxa = len(pres_df)
    num_ecoregions = len(list(list(pres_df.values())[0].values())[0][econame].unique())
    num_rows = num_models * num_taxa *2 * num_ecoregions # the two is one for test, one for train
    score_name = [s.__name__ for s in mets]
    extra = ['model', 'taxa', 'test', 'ecoregion'] 
    score_idx = {score : (i + len(extra)) for score, i in zip(score_name, range(len(score_name)))}
    df_names = extra + score_name
    model_idx, taxa_idx, test_idx, eco_idx = 0, 1, 2, 3
    filler = np.full([num_rows, len(df_names)], -1)
    results_df_reg = pd.DataFrame(filler, columns=df_names)

    # fill in dataframe
    i = 0
    # one entry per-taxa
    for taxa, dic in pres_df.items():
        # one entry per-model
        for name, dff in dic.items():
            for ecoregion, df in dff.groupby(econame):

                col_2_keep = list(taxa_names[taxa])
                gt = ground_truth[taxa]
                # test
                curr_df =  df[df.test]
                obs = curr_df[col_2_keep].values
                # if no observations in the given ecoregion, move on
                if len(obs) < 1:
                    continue            
                curr_gt = gt[gt.test]
                curr_gt = curr_gt[curr_gt[econame] == ecoregion]
                curr_yt =  curr_gt[col_2_keep].values
                results_df_reg = run_metrics(results_df_reg, curr_yt, obs, mets, score_idx, i)
                results_df_reg.iloc[i, taxa_idx] = taxa
                results_df_reg.iloc[i, test_idx] = 'test'
                results_df_reg.iloc[i, model_idx] = name
                results_df_reg.iloc[i, eco_idx] = ecoregion
                i += 1
                # copying code, bad but here we are..
                # train
                curr_df =  df[~df.test]
                obs = curr_df[col_2_keep].values
                if len(obs) < 1:
                    continue                 
                curr_gt = gt[~gt.test]
                curr_gt = curr_gt[curr_gt[econame] == ecoregion]            
                curr_yt =  curr_gt[col_2_keep].values
                results_df_reg = run_metrics(results_df_reg, curr_yt, obs, mets, score_idx, i)
                results_df_reg.iloc[i, taxa_idx] = taxa
                results_df_reg.iloc[i, test_idx] = 'train'
                results_df_reg.iloc[i, model_idx] = name
                results_df_reg.iloc[i, eco_idx] = ecoregion
                i += 1
    return results_df_reg


# 3. per-species statistics
# have one df for each taxa
# model  support test metric  sp1  sp2 ...
# random   T       T   f1      .2   .23
# add one line for each of the above 

# gotta manually go taxa at a time
def sklearn_per_taxa_individual(yo, yt, taxa_names):

        # one entry per-model
    num_models = len(yo.keys())
    num_rows = num_models  *2 * 4 # the two is one for test, one for train
    score_name = ['metric', 'model', 'support', 'test']
    col_2_keep = list(taxa_names)
    df_names = score_name + col_2_keep
    metric_idx, model_idx, support_idx, test_idx = 0, 1, 2, 3
    filler = np.full([num_rows, len(df_names)], 'NaNNANANANANANANANANAANANANANANANANAN')

    yt_test = yt[yt.test]
    yt_test = yt_test[col_2_keep]
    yt_test = yt_test.values
    yt_train = yt[~yt.test]
    yt_train = yt_train[col_2_keep]
    yt_train - yt_train.values
    i = 0
    for name, df in yo.items():

        # test
        obs_test = df[df.test]
        obs_test = obs_test[col_2_keep].values
        obs_train = df[~df.test]
        obs_train = obs_train[col_2_keep].values    

        tick = time.time()

        precision, recall, fbeta, support = metrics.precision_recall_fscore_support(yt_test, obs_test)
        # precision
        filler[i, metric_idx] = 'precision'
        filler[i, model_idx] = name
        filler[i, support_idx] = False
        filler[i, test_idx] = 'test'        
        filler[i, (test_idx+1):] = precision
        i += 1
        #recalll
        filler[i, metric_idx] = 'recall'
        filler[i, model_idx] = name
        filler[i, support_idx] = False
        filler[i, test_idx] =  'test'
        filler[i, (test_idx+1):] = recall    
        i += 1    
        #fbeta
        filler[i, metric_idx] = 'fbeta'
        filler[i, model_idx] = name
        filler[i, support_idx] = False
        filler[i, test_idx] =  'test'
        filler[i, (test_idx+1):] = fbeta    
        i += 1
        #support
        filler[i, metric_idx] = 'support'
        filler[i, model_idx] = name
        filler[i, support_idx] = True
        filler[i, test_idx] =  'test'
        filler[i, (test_idx+1):] = support
        i += 1

        # train
        precision, recall, fbeta, support = metrics.precision_recall_fscore_support(yt_train, obs_train)
        # precision
        filler[i, metric_idx] = 'precision'
        filler[i, model_idx] = name
        filler[i, support_idx] = False
        filler[i, test_idx] =  'train'
        filler[i, (test_idx+1):] = precision
        i += 1
        #recalll
        filler[i, metric_idx] = 'recall'
        filler[i, model_idx] = name
        filler[i, support_idx] = False
        filler[i, test_idx] = 'train'
        filler[i, (test_idx+1):] = recall    
        i += 1    
        #fbeta
        filler[i, metric_idx] = 'fbeta'
        filler[i, model_idx] = name
        filler[i, support_idx] = False
        filler[i, test_idx] = 'train'
        filler[i, (test_idx+1):] = fbeta    
        i += 1
        #support
        filler[i, metric_idx] = 'support'
        filler[i, model_idx] = name
        filler[i, support_idx] = True
        filler[i, test_idx] = 'train'
        filler[i, (test_idx+1):] = support
        i += 1    
        tock = time.time()
        print("{} took {} minutes".format(name, ((tock-tick)/60)))                

        # TODO: move this to the end
    per_spec_df = pd.DataFrame(filler, columns=df_names)    
    return per_spec_df




# 3. per-observation statistics
# have one df for each taxa
# model  support test metric  sp1  sp2 ...
# random   T       T   f1      .2   .23
# add one line for each of the above 
# one entry per-model
def inhouse_per_observation(yo, yt, taxa_names, device):
    num_models = len(yo.keys())
    num_obs = len(list(yo.values())[0])
    num_rows = num_models  * num_obs * 3 # 3 is for recall, precision, f1
    score_name = ['metric', 'model', 'test', 'value', 'latitude', 'longitude']
    metric_idx, model_idx, test_idx, value_idx, lat_idx, lon_idx = 0, 1, 2, 3, 4, 5
    df_names = score_name 
    # this is really bad code but essentially to store as numpy array need to 
    # create numpy array as object and the longer the string the more digits you get yikes
    filler = np.full([num_rows, len(df_names)], 'NANANANANANANANANANANANAN')
    df = list(yo.values())[0]
    test_names = ['test' if n else 'train' for n in df.test]
    lats = df.lat.tolist()
    lons = df.lon.tolist()
    yt = yt[list(taxa_names)]
    yt = yt.values
    yt_t = torch.tensor(yt)
    yt_t = yt_t.to(device)

    i = 0
    for name, df in yo.items():

        tick = time.time()
        modelname = [name] * num_obs
        obs = df[list(taxa_names)]
        obs = obs.values
        obs_t = torch.tensor(obs)
        obs_t = obs_t.to(device)
        ans = utils.precision_per_obs(obs_t.float(), yt_t.float(), device=device)
        ans = ans.cpu().numpy()
        metricname = ['precision'] * num_obs
        filler[i:i+num_obs, model_idx] = modelname
        filler[i:i+num_obs, metric_idx] = metricname
        filler[i:i+num_obs, test_idx] = test_names
        filler[i:i+num_obs, lat_idx] = lats    
        filler[i:i+num_obs, lon_idx] = lons        
        filler[i:i+num_obs, value_idx] = ans

        i = i + num_obs

        ans = utils.recall_per_obs(obs_t.float(), yt_t.float(), device=device)
        metricname = ['recall'] * num_obs
        filler[i:i+num_obs, model_idx] = modelname
        filler[i:i+num_obs, metric_idx] = metricname
        filler[i:i+num_obs, test_idx] = test_names    
        filler[i:i+num_obs, lat_idx] = lats    
        filler[i:i+num_obs, lon_idx] = lons    
        filler[i:i+num_obs, value_idx] = ans.cpu().numpy()
        i = i + num_obs

        ans = utils.f1_per_obs(obs_t.float(), yt_t.float(), device=device)
        metricname = ['f1'] * num_obs
        filler[i:i+num_obs, model_idx] = modelname
        filler[i:i+num_obs, metric_idx] = metricname
        filler[i:i+num_obs, test_idx] = test_names    
        filler[i:i+num_obs, lat_idx] = lats    
        filler[i:i+num_obs, lon_idx] = lons    
        filler[i:i+num_obs, value_idx] = ans.cpu().numpy()
        i = i + num_obs
        tock = time.time()
        print("{} took {} minutes".format(name, ((tock-tick)/60)))                

    res_df = pd.DataFrame(filler, columns=df_names)    
    return res_df



        
        
if __name__ == "__main__":
    
    args = ['base_dir', 'pres_threshold', 'device', 'config_path', 'ecoregion', 'num_species', 'which_taxa']
    ARGS = config.parse_known_args(args)       
    print(args)
    run_metrics_and_longform(ARGS)
    
