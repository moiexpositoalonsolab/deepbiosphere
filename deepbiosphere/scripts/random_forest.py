from datetime import datetime
import time
import deepbiosphere.scripts.GEOCLEF_Run as run
import pandas as pd
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
    df_spec = pd.DataFrame(total_spec)
    df_gen = pd.DataFrame(total_gen)
    df_fam = pd.DataFrame(total_fam)    
    inv_gen = {v: k for k, v in dset.gen_dict.items()}
    inv_fam = {v: k for k, v in dset.fam_dict.items()}
    df_spec.columns = [dset.inv_spec[i] for i in range(dset.num_specs)]
    df_gen.columns = [inv_gen[i] for i in range(dset.num_gens)]
    df_fam.columns = [inv_fam[i] for i in range(dset.num_fams)]
    to_transfer = ['lat', 'lon', 'region', 'city', 'NA_L3NAME', 'US_L3NAME', 'NA_L2NAME', 'NA_L1NAME', 'test']
    df_spec[to_transfer] = obs[to_transfer]
    df_gen[to_transfer] = obs[to_transfer]
    df_fam[to_transfer] = obs[to_transfer]    
    
    if num_species < 0:
        nsp = 'all_spec'
    else:
        nsp = "top_{}_spec".format(num_species)    
    # get good name and save to csv
    name_spec = "{}_{}_{}_{}_{}_{}_{}.csv".format("RandomForestClassifier", params.params.n_trees ,  'species' , nsp, datetime.now().day, datetime.now().month, datetime.now().year)
    name_gen = "{}_{}_{}_{}_{}_{}_{}.csv".format("RandomForestClassifier", params.params.n_trees, 'genera' , nsp, datetime.now().day, datetime.now().month, datetime.now().year)
    name_fam = "{}_{}_{}_{}_{}_{}_{}.csv".format("RandomForestClassifier", params.params.n_trees, 'family' , nsp, datetime.now().day, datetime.now().month, datetime.now().year,)
    df_spec.to_csv(base_dir + '/inference/' + name_spec)
    df_gen.to_csv(base_dir + '/inference/' +name_gen)
    df_fam.to_csv(base_dir + '/inference/' + name_fam)       


    Y_spec = Y[:,:dset.num_specs]
    Y_gen = Y[:,dset.num_specs: (dset.num_specs + dset.num_gens)]
    Y_fam = Y[:, (dset.num_specs + dset.num_gens):]
    assert Y_spec.shape == (len(dset), dset.num_specs)
    assert Y_gen.shape == (len(dset), dset.num_gens)
    assert Y_fam.shape == (len(dset), dset.num_fams)    
    ytru_spec = pd.DataFrame(Y_spec)
    ytru_gen  = pd.DataFrame(Y_gen)
    ytru_fam  = pd.DataFrame(Y_fam)
    
    ytru_spec.columns = [dset.inv_spec[i] for i in range(dset.num_specs)]
    ytru_gen.columns = [inv_gen[i] for i in range(dset.num_gens)]
    ytru_fam.columns = [inv_fam[i] for i in range(dset.num_fams)]
    to_transfer = ['lat', 'lon', 'region', 'city', 'NA_L3NAME', 'US_L3NAME', 'NA_L2NAME', 'NA_L1NAME', 'test']
    ytru_spec[to_transfer] = obs[to_transfer]
    ytru_gen[to_transfer] = obs[to_transfer]
    ytru_fam[to_transfer] = obs[to_transfer]        
    

    # get good name and save to csv
    name_spec = "ytrue_species_{}.csv".format(nsp)
    name_gen = "ytrue_genus_{}.csv".format(nsp)
    name_fam = "ytrue_family_{}.csv".format(nsp)    
    ytru_spec.to_csv(base_dir + '/inference/' + name_spec)
    ytru_gen.to_csv(base_dir + '/inference/' + name_gen)
    ytru_fam.to_csv( base_dir + '/inference/' + name_fam)       
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
    
    ARGS = config.parse_known_args(args)
    random_forest(params, ARGS.base_dir, ARGS.num_species, ARGS.processes)
