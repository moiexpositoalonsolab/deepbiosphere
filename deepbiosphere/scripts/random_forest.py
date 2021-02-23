from datetime import datetime
import time
import deepbiosphere.scripts.GEOCLEF_Run as run
import pandas as pd
from types import SimpleNamespace 
from deepbiosphere.scripts.GEOCLEF_Config import paths, Run_Params
import deepbiosphere.scripts.GEOCLEF_Config as config
# Import train_test_split function
from sklearn.model_selection import train_test_split
from deepbiosphere.scripts import GEOCLEF_Dataset as dataset
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn import metrics

    
def random_forest(params, base_dir, num_species, processes):
    
    dset= run.setup_dataset(params.params.observation, params.base_dir, params.params.organism, params.params.region, params.params.normalize, params.params.no_altitude, params.params.dataset, params.params.threshold, num_species=num_species)
    _, _, idxs = run.better_split_train_test(dset)
    obs = dataset.get_gbif_observations(base_dir, params.params.organism, params.params.region, params.params.observation, params.params.threshold, num_species)
    obs.fillna('nan', inplace=True)
    if 'species' not in obs.columns:
        obs = utils.add_taxon_metadata(self.base_dir, obs, self.organism)
    tick = time.time()
    X = np.zeros((len(dset), dset.num_rasters))
    Y = np.zeros((len(dset), dset.num_specs+dset.num_gens+dset.num_fams))
    
    for i, stuff in enumerate(dset):

        (all_spec, all_gen, all_fam, env_rasters, _) = stuff

        specs = np.zeros(dset.num_specs)
        specs[all_spec] += 1
        gens = np.zeros(dset.num_gens)
        gens[all_gen] += 1
        fams = np.zeros(dset.num_fams)
        fams[all_fam] += 1
        X[i,:] = env_rasters
        combined = np.concatenate([specs, gens, fams], axis=0)
        Y[i,:] = combined
    tock = time.time()
    print("took ", tock-tick, " seconds to build dataset")
    
    

    Xtest, Xtrain = X[idxs['test']], X[idxs['train']]
    Ytest, Ytrain = Y[idxs['test']], Y[idxs['train']]
    
    tick = time.time()
    if params.params.unweighted:
        class_weight = 'None'
    else:
        class_weight = 'balanced'
    clf = RandomForestClassifier(n_estimators=params.params.n_trees, verbose=3, n_jobs=processes, class_weight = class_weight)
    clf = clf.fit(Xtrain, Ytrain)
    tock = time.time()
    print("took ", (tock-tick)/60, " minutes to train Joint RFC")
    
    # predict on train
    tick = time.time()
    pred = clf.predict_proba(Xtest)
    res = np.full([len(pred[0]), len(pred)], np.nan)
    for  i, sp in enumerate(pred):
        ob = range(len(sp))
        res[:, i] = sp[:, 1]
    test_spec = res[:,:dset.num_specs]
    test_gen = res[:,dset.num_specs: (dset.num_specs + dset.num_gens)]
    test_fam = res[:, (dset.num_specs + dset.num_gens):]
    assert test_spec.shape[1] == dset.num_specs
    assert test_gen.shape[1] == dset.num_gens
    assert test_fam.shape[1] == dset.num_fams

    tock = time.time()
    print("took ", (tock-tick)/60, " minutes to predict on test set of Joint RFC")
    # predict on test
    tick = time.time()
    pred = clf.predict_proba(Xtrain)
    res = np.full([len(pred[0]), len(pred)], np.nan)
    for  i, sp in enumerate(pred):
        ob = range(len(sp))
        res[:, i] = sp[:, 1]
    train_spec = res[:,:dset.num_specs]
    train_gen = res[:,dset.num_specs: (dset.num_specs + dset.num_gens)]
    train_fam = res[:, (dset.num_specs + dset.num_gens):]
    assert train_spec.shape[1] == dset.num_specs
    assert train_gen.shape[1] == dset.num_gens
    assert train_fam.shape[1] == dset.num_fams
        
    tock = time.time()
    print("took ", (tock-tick)/60, " minutes to predict on train set of Joint RFC")    
    total_spec = np.full([len(dset), dset.num_specs], np.nan)
    total_spec[idxs['test']] = test_spec
    total_spec[idxs['train']] = train_spec    
    
    total_gen = np.full([len(dset), dset.num_gens], np.nan)
    total_gen[idxs['test']] = test_gen
    total_gen[idxs['train']] = train_gen    
    
    total_fam = np.full([len(dset), dset.num_fams], np.nan)
    total_fam[idxs['test']] = test_fam
    total_fam[idxs['train']] = train_fam
    print("saving data")
    tick = time.time()
    to_transfer = ['lat', 'lon', 'region', 'city', 'NA_L3NAME', 'US_L3NAME', 'NA_L2NAME', 'NA_L1NAME', 'test']    
    inv_gen = {v: k for k, v in dset.gen_dict.items()}
    inv_fam = {v: k for k, v in dset.fam_dict.items()}
    spec_cols = [dset.inv_spec[i] for i in range(dset.num_specs)]
    gen_cols = [inv_gen[i] for i in range(dset.num_gens)]
    fam_cols = [inv_fam[i] for i in range(dset.num_fams)]    
    
    df_spec = utils.numpy_2_df(total_spec, spec_cols, obs, to_transfer)
    df_gen  = utils.numpy_2_df(total_gen, gen_cols, obs, to_transfer)
    df_fam  = utils.numpy_2_df(total_fam, fam_cols, obs, to_transfer)
    pth_spec = config.build_inference_path(base_dir, params.params.model, params.params.loss, params.params.exp_id, 'species', num_species)
    pth_gen = config.build_inference_path(base_dir, params.params.model, params.params.loss, params.params.exp_id, 'genus', num_species)
    pth_fam = config.build_inference_path(base_dir, params.params.model, params.params.loss, params.params.exp_id, 'family', num_species)
    df_spec.to_csv(pth_spec)
    df_gen.to_csv(pth_gen)
    df_fam.to_csv(pth_fam)


    Y_spec = Y[:,:dset.num_specs]
    Y_gen = Y[:,dset.num_specs: (dset.num_specs + dset.num_gens)]
    Y_fam = Y[:, (dset.num_specs + dset.num_gens):]
    assert Y_spec.shape == (len(dset), dset.num_specs)
    assert Y_gen.shape == (len(dset), dset.num_gens)
    assert Y_fam.shape == (len(dset), dset.num_fams)
    
    # create config file for y-trues

    paramss = {
        'lr': 'none',
        'observation': params.params.observation,
        'organism' : params.params.organism,
        'region' : params.params.region,
        'model' : 'Ground_truth',
        'exp_id' : params.params.exp_id,
        'seed' : 'none',
        'batch_size' : 'none',
        'loss' : 'none',
        'normalize' : 'none',
        'unweighted' : 'none',
        'no_alt' : 'none',
        'dataset' : 'none',
        'threshold' : 'none',
        'loss_type' : 'none',
        'pretrained' : 'none',
        'batch_norm' : 'none',
        'arch_type' : 'none',
        'load_from_config' : None,
        'base_dir' : base_dir
    }

    paramss = SimpleNamespace(**paramss)
    ytrues = config.Run_Params(base_dir, paramss)
    ytru_spec = utils.numpy_2_df(Y_spec, spec_cols, obs, to_transfer)
    ytru_gen  = utils.numpy_2_df(Y_gen, gen_cols, obs, to_transfer)
    ytru_fam  = utils.numpy_2_df(Y_fam, fam_cols, obs, to_transfer)
    pth_spec = config.build_inference_path(base_dir, paramss.params.model, paramss.params.loss, paramss.params.exp_id, 'species', num_species)
    pth_gen = config.build_inference_path(base_dir, paramss.params.model, paramss.params.loss, paramss.params.exp_id, 'genus', num_species)
    pth_fam = config.build_inference_path(base_dir, paramss.params.model, paramss.params.loss, paramss.params.exp_id, 'family', num_species)
    ytru_spec.to_csv(pth_spec)
    ytru_gen.to_csv(pth_gen)
    ytru_fam.to_csv(pth_fam)
    tock = time.time()
    print("took {} minutes to save data".format((tock-tick)/60))
    
if __name__ == "__main__":
    # add CLI to nuke a whole model
    # add CLI to clean up all model files

    args = ['base_dir', 'num_species', 'observation', 'organism', 'region', 'exp_id', 'seed', 'normalize', 'unweighted', 'dataset', 'threshold', 'model', 'load_from_config', 'loss', 'no_alt', 'n_trees', 'processes']
    # hacky set the model to be RandomForestCLassifier
    ARGS = config.parse_known_args(args)       
#     ARGS['model'] = 'RandomForestClassifier'
    config.setup_main_dirs(ARGS.base_dir)
    params = config.Run_Params(ARGS.base_dir, ARGS)
    # TODO: make sure you can only set model to be random forest here    
    ARGS = config.parse_known_args(args)
    random_forest(params, ARGS.base_dir, ARGS.num_species, ARGS.processes)
