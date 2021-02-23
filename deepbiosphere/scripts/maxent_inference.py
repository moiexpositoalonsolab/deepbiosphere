import time
import glob
import rasterio
import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
import deepbiosphere.scripts.GEOCLEF_Run as run
import deepbiosphere.scripts.GEOCLEF_Utils as utils
import deepbiosphere.scripts.GEOCLEF_Dataset as dataset
from deepbiosphere.scripts import GEOCLEF_Config as config
from deepbiosphere.scripts.GEOCLEF_Config import paths, Run_Params
from deepbiosphere.scripts.GEOCLEF_Run import  setup_dataset, setup_model, setup_loss

def maxent_inference(base_dir, params, num_species):
    
    print("getting data")
    # TODO: make sure dataframe has all the info it needs for plotting
    obs = dataset.get_gbif_observations(base_dir, params.params.organism, params.params.region, params.params.observation, params.params.threshold, num_species)
    obs.fillna('nan', inplace=True)
    if 'species' not in obs.columns:
        obs = utils.add_taxon_metadata(self.base_dir, obs, self.organism)

    dset= run.setup_dataset(params.params.observation, params.base_dir, params.params.organism, params.params.region, params.params.normalize, params.params.no_altitude, params.params.dataset, params.params.threshold, num_species=num_species)
    train_samp, test_samp, idxs = run.better_split_train_test(dset)
    # load in tiffs as rasters
    
    # TODO: make cleaner
    rasnames = f"{paths.DBS_DIR}occurrences/MaxentResults_All/*.tif"
    files = glob.glob(rasnames)
    all_ras = []
    for file in files:
        src = rasterio.open(file)
        specname = file.split('/')[-1].split('_Proj')[0].replace('_', ' ')
        temp = src.read().squeeze()
        nodata = temp.min()
        # trick: convert nan value points to 0 probability!
        # if nodata < 0.0:
#             print("setting nan value for {} from {} to 0.0".format(specname, nodata, " to 0.0"))
        temp[temp == nodata] = 0.0
        all_ras.append((specname, src.transform, temp))    

    print("extracting predictions")
    tick = time.time()
    # so I think the negative values are because the rasters are only fit around the species range, but to certify that what I'll do is plot the rasters + the offending point with geopandas
    maxent_pred = np.full([len(obs), dset.num_specs], np.nan)
    maxent_gen = np.full([len(obs), dset.num_gens], np.nan)
    maxent_fam = np.full([len(obs), dset.num_fams], np.nan)
    sp_2_gen = utils.dict_from_columns(obs, 'species', 'genus')
    sp_2_fam = utils.dict_from_columns(obs, 'species', 'family')    
    to_iterate = dset.obs[:, dataset.lat_lon_idx].tolist()
    # TODO: rewrite this order so it's faster
    # loop over rasters, not indices because you can

        # order species in order expected for file
            # need (rasters, spec_name_same)
    for spec, trans, raster in all_ras:
        spc_idx = dset.spec_dict[spec]
        gen_idx = dset.gen_dict[sp_2_gen[spec]]
        fam_idx = dset.fam_dict[sp_2_fam[spec]]
        for i, (lat,lon) in enumerate(to_iterate):

#             print("valid idx? ", spc_idx)
            x, y = dataset.latlon_2_idx(trans, (lat, lon))
            if x < 0 or y < 0 or x >= raster.shape[0] or y >= raster.shape[1]:
                # this means that maxent predicted no probability in this area, so the raster is cut off for this region
                # so can go ahead and say 0 probability
                maxent_pred[i, spc_idx] = 0.0
                maxent_gen[i, gen_idx] = 0.0
                maxent_fam[i, fam_idx] = 0.0                
            else:
                # convert species to genus, family
                maxent_pred[i, spc_idx] = raster[x,y]
                maxent_gen[i, gen_idx] = raster[x,y]
                maxent_fam[i, fam_idx] = raster[x,y]

            
    tock = time.time()
    print("extracting predictions took {} minutes".format((tock-tick)/60))
    #  check how many nans are left, if reasonable amount then just convert to 0.0
    num_nans = maxent_pred[maxent_pred == np.nan]
    print("num nans is ", num_nans.shape)
    # convert to pandas dataframe and save in the correct location
    print('saving data')
    tick = time.time()
    to_transfer = ['lat', 'lon', 'region', 'city', 'NA_L3NAME', 'US_L3NAME', 'NA_L2NAME', 'NA_L1NAME', 'test']
    inv_gen = {v: k for k, v in dset.gen_dict.items()}
    inv_fam = {v: k for k, v in dset.fam_dict.items()}
    
    df_spec_cols = [dset.inv_spec[i] for i in range(dset.num_specs)]
    df_gen_cols = [inv_gen[i] for i in range(dset.num_gens)]
    df_fam_cols = [inv_fam[i] for i in range(dset.num_fams)]    

    df_spec = utils.numpy_2_df(maxent_pred, df_spec_cols, obs, to_transfer)
    df_gen = utils.numpy_2_df(maxent_gen, df_gen_cols, obs, to_transfer)
    df_fam = utils.numpy_2_df(maxent_fam, df_fam_cols, obs, to_transfer)
    pth_spec = config.build_inference_path(base_dir, params.params.model, params.params.loss, params.params.exp_id, 'species', num_species)
    pth_gen = config.build_inference_path(base_dir, params.params.model, params.params.loss, params.params.exp_id, 'genus', num_species)
    pth_fam = config.build_inference_path(base_dir, params.params.model, params.params.loss, params.params.exp_id, 'family', num_species)
    
    df_spec.to_csv(pth_spec)
    df_gen.to_csv(pth_gen)
    df_fam.to_csv(pth_fam)
    tock = time.time()
    print("took {} minutes to save data".format((tock-tick)/60))

    

    
        
# TODO: see if can embed R into this and run the maxent??
# yes! can use rpy2!
def train_maxent():
    # 1. get 
    # robjects.r('install.packages("rJava")')
    to_import = ['devtools', 'rJava', 'dismo', 'raster', 'foreach', 'doParallel', 'sp']
    
#     install.packages("rJava")
# devtools::install_github("s-u/rJava")
#     install.packages("dismo")
    pckgs = {}
    for imp in to_import:
        pckgs[imp] = importr(imp)
    print(pckgs)
    
if __name__ == "__main__":
    np.testing.suppress_warnings()

    args = ['base_dir', 'num_species', 'observation', 'organism', 'region', 'exp_id', 'seed', 'normalize', 'dataset', 'threshold', 'model', 'load_from_config', 'loss', 'no_alt']

    ARGS = config.parse_known_args(args)
    config.setup_main_dirs(ARGS.base_dir)
    params = config.Run_Params(ARGS.base_dir, ARGS)
    # TODO: make sure you can only set model to be maxent here
    ARGS = config.parse_known_args(args)
#     train_maxent()
    maxent_inference(ARGS.base_dir, params, ARGS.num_species)

