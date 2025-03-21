# rasterio packages
import rasterio
from rasterio import merge
from rasterio import Affine
from rasterio import profiles
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling

# deepbiosphere packages
import deepbiosphere.Run as run
import deepbiosphere.Utils as utils
import deepbiosphere.Models as mods
from deepbiosphere.Utils import paths
import deepbiosphere.Dataset as dataset
import deepbiosphere.NAIP_Utils as naip
import deepbiosphere.Build_Data as build
import deepbiosphere.Losses as losses

# GIS packages
import shapely as shp
import geopandas as gpd
from pyproj import Transformer as projTransf
from shapely.geometry import Point, Polygon

# ML + statistics packages
import torch
import numpy as np
import pandas as pd
from scipy.spatial import distance
import torchvision.transforms.functional as TF

# miscellaneous packages
import os
import csv
import sys
import glob
import time
import json
import math
import warnings
import argparse
import multiprocessing
from tqdm import tqdm
from typing import List, Tuple
from datetime import date
from types import SimpleNamespace


# ---------- Merging and saving predictions made per-species ---------- #



# ---------- Calculating alpha, beta diversity etc. post-hoc ---------- #

def calculate_extra_attributes_parallel(procid,
                                        lock,
                                        files, 
                                        save_dir,
                                        pred_types,
                                        alpha_type,
                                       pred_specs : List[str],
                                       overwrite : bool =False):
    
    with lock:
        prog = tqdm(total=len(files), desc=f"Extra attributes tiff group #{procid}", unit=' tiffs', position=procid)
    for file in files:
        calculate_extra_attributes(file, save_dir,pred_types,alpha_type, pred_specs, overwrite)
        with lock:
            prog.update(1)
    with lock:
        prog.close()
    
            
def calculate_extra_attributes(file, 
                               save_dir,
                               pred_types,
                               alpha_type,
                              pred_specs : List[str],
                              overwrite=False):

    rfile = rasterio.open(file)
    spec_names = rfile.descriptions
    n_specs = len(spec_names)
    pred = rfile.read()
    pred_aff = rfile.transform
    out_res = rfile.res
    out_bounds = rfile.bounds
    subdir = file.split('/')[-2]
    save_dir = f"{save_dir}{subdir}/"
    save_name = f"{file.split('/')[-1].split('_raw_spec')[0]}"
    # typecheck predictions parameter
    pred_types = [naip.Prediction[pred_type] for pred_type in pred_types]
    alpha_type = naip.Alpha[alpha_type]
    # and save out predictions
    for pred_type in pred_types:
        # first check that file hasn't been generated
        if not overwrite:
            if naip.check_file_exists(save_dir, save_name, pred_type.value):
                continue
                
        if pred_type is naip.Prediction.PER_SPEC:
            naip.save_tiff_per_species(save_dir=save_dir,
                                      save_name=save_name,
                                      preds=pred,
                                      transf=pred_aff,
                                      crs=rfile.crs,
                                      null_val=np.nan,
                                      out_res=out_res,
                                      bounds=out_bounds,
                                      spec_names=spec_names,
                                      pred_specs=pred_specs,
                                      overwrite=overwrite)
            
                    
            
        # Generate alpha diversity from species predictions
        elif pred_type is naip.Prediction.ALPHA:
            # prevent modification of
            # underlying pred array
            a_pred = np.copy(pred)
            alpha_pred = naip.predict_alpha(a_pred, alpha_type)
            alpha_pred = np.expand_dims(alpha_pred, axis=0)
            naip.save_tiff(
                save_dir=save_dir,
                save_name=save_name,
                pred_type=naip.Prediction.ALPHA.value,
                preds=alpha_pred,
                transf=pred_aff,
                crs=rfile.crs,
                null_val=-1 if alpha_type is naip.Alpha.THRES else np.nan,
                out_res=out_res,
                bounds=out_bounds,
                band_names=[naip.Prediction.ALPHA.value])

        # Generate beta diversity from species predictions
        elif pred_type is naip.Prediction.BETA:
            # prevent modification of
            # underlying pred array
            b_pred = np.copy(pred)
            beta_pred, beta_aff, beta_res, beta_bounds = naip.predict_beta(b_pred, pred_aff)
            beta_pred= np.expand_dims(beta_pred, axis=0)
            # and save out
            naip.save_tiff(
                save_dir=save_dir,
                save_name=save_name,
                pred_type=naip.Prediction.BETA.value,
                preds=beta_pred,
                transf=beta_aff,
                crs=rfile.crs,
                null_val=np.nan,
                out_res=beta_res,
                bounds=beta_bounds,
                band_names=[naip.Prediction.BETA.value])
    rfile.close()

def additional_attributes_pertiff(parent_dir : str,
                                pred_res : int,
                                pred_year : int,
                                band : int,
                                exp_id : str,
                                epoch : int,
                                processes : int,
                                pred_types : List[str] = ['RAW'], 
                                alpha_type : str = 'SUM',
                                resolution : int = utils.IMG_SIZE,
                                pred_specs : List[str]=None,
                                overwrite : bool =False):

    # get raw predictions from file
    save_dir = f"{paths.RASTERS}{parent_dir}/{pred_res}m_{pred_year}_{band}_{exp_id}_{epoch}/"
    files = glob.glob(f"{save_dir}*/*raw_spec.tif")
    # otherwise read it from the open tiff
    # now get the actual predictions
    if processes > 1:
        files = utils.partition(files, processes)
        lock = multiprocessing.Manager().Lock()
        pool =  multiprocessing.Pool(processes)

        res_async = [pool.apply_async(calculate_extra_attributes_parallel, args=(i, lock, file, save_dir, pred_types, alpha_type, pred_specs, overwrite)) for i, file in enumerate(files)]
        res_files = [r.get() for r in res_async]
        pool.close()
        pool.join()
    else:
        for file in tqdm(files, total=len(files), desc=f"Extra attributes ", unit=' tiffs'):
            calculate_extra_attributes(file=file, 
                                       save_dir=save_dir,
                                       pred_types=pred_types,
                                       alpha_type=alpha_type,
                                       pred_specs=pred_specs,
                                      overwrite=overwrite)
    
        
    
# ---------- driver code ---------- #

def predict_rasters_serial(rasters : List[str], 
                           parent_dir : str,
                           model_config : SimpleNamespace,
                           device_no : int,
                           batch_size : int,
                           epoch : int,
                           pred_year : int, 
                           pred_types : List[str] = ['RAW'],
                           alpha_type : str = 'SUM',
                           resolution : int = utils.IMG_SIZE,
                           img_size : int = utils.IMG_SIZE,
                           impute_climate : bool = True,
                           sat_res : int = None,
                           clim_rasters : List = None,
                           pred_specs : List[str] = None):
    if device_no < 0 :
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device_no}")
        torch.cuda.set_device(device_no)
    if resolution < 20:
        raise ValueError("resolution needs to be at least 20 for model to work!")

    model = run.load_model(device, model_config, epoch, logging=False) 
    model = model.to(device)
    # don't forget eval mode
    model.eval()

    results = []
    for raster in tqdm(rasters, unit='tiff', desc=f"{pred_year} tiffs"):
        # get the subdirectory where to save this image
        subdir = raster.split('/')[-2]
        save_dir = f"{parent_dir}{subdir}/"
        save_name = f"{raster.split('/')[-1].split('.')[0]}_{str(pred_year)}"
        raster = rasterio.open(raster)
        # make subdir if it doesn't exist
        # if the parent directory doesn't 
        # exist, will fail to prevent creating
        # a directory in an unwanted place
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        # and now predict that raster
        results.append(naip.predict_raster(raster, 
                       save_dir, 
                       save_name, 
                       model, 
                       model_config, 
                       device, 
                       batch_size, 
                       pred_types, 
                       alpha_type,
                       resolution, 
                       img_size, 
                       impute_climate, 
                       clim_rasters=clim_rasters,
                       sat_res = sat_res,
                       disable_tqdm = True,
                       pred_specs =pred_specs))
    return results
    
    
## parallelized version of taking list of rasters and splitting up
# with parallel implemented etc.
def predict_rasters_parallel(procid : int,
                             lock : multiprocessing.Manager,
                             rasters : List[str], 
                             parent_dir : str,
                             model_config : SimpleNamespace,
                             device_no : int,
                             batch_size : int,
                             epoch : int,
                             pred_year : int,
                             pred_types : List[str] = ['RAW'],
                             alpha_type : str = 'SUM',
                             resolution : int = utils.IMG_SIZE,
                             sat_res : int = None,
                             impute_climate : bool = True,
                             clim_rasters : List = None,
                             specs : List[str] = None,
                             img_size : int = utils.IMG_SIZE,
                             pred_specs : List[str] = None):


    if device_no < 0 :
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device_no}")
        torch.cuda.set_device(device_no)
    if resolution < 20:
        raise ValueError("resolution needs to be at least 20 for model to work!")

    model = run.load_model(device, model_config, epoch, logging=False) 
    model = model.to(device)
    # don't forget eval mode
    model.eval()

    # set up TQDM for parallel
    with lock:
        prog = tqdm(total=len(rasters), desc=f"{pred_year} tiffs group #{procid}", unit=' tiffs', position=procid)
    files = []  
    for i, raster in enumerate(rasters):
        # get the subdirectory where to save this image
        subdir = raster.split('/')[-2]
        save_dir = f"{parent_dir}{subdir}/"
        save_name = f"{raster.split('/')[-1].split('.')[0]}_{str(pred_year)}"
        raster = rasterio.open(raster)
        # make subdir if it doesn't exist
        # if the parent directory doesn't 
        # exist, will fail to prevent creating
        # a directory in an unwanted place
        if not os.path.isdir(save_dir):
            os.mkdir(save_dir)
        # and now predict that raster
        files.append(naip.predict_raster(raster, 
                       save_dir, 
                       save_name, 
                       model, 
                       model_config, 
                       device, 
                       batch_size, 
                       pred_types, 
                       alpha_type,
                       resolution,
                       img_size, 
                       impute_climate, 
                       clim_rasters=clim_rasters,
                       disable_tqdm = True, 
                       sat_res = sat_res,
                       pred_specs = pred_specs))
        with lock:
            prog.update(1)
    with lock:
        prog.close()
    

def predict_rasters_list(pred_outline : gpd.GeoDataFrame,
                         pred_types : List[str],
                         alpha_type : str,
                         parent_dir : str, #
                         cfg : SimpleNamespace,
                         epoch : int,
                         band : int,
                         pred_year : int,
                         state : str,
                         device: int,
                         n_processes : int, # whether to use parallel or not
                         batch_size : int,
                         pred_res: int, # meter resolution of predictions to make with DBS
                         sat_res : int = None, # resolution to upsample sat imagery to
                         impute_climate = True,
                         clim_rasters = None,
                         pred_specs : List[str] = None):

    # type check
    pred_types = [naip.Prediction[pred_type] for pred_type in pred_types]
    save_dir = f"{paths.RASTERS}{parent_dir}/{pred_res}m_{pred_year}_{band}_{cfg.exp_id}_{epoch}/"
    # prep bounds
    naip_shp = naip.load_naip_bounds(paths.SHPFILES, state, pred_year)
    pred_outline = pred_outline.to_crs(naip_shp.crs)# TODO: Return this!! .dissolve()
    # messy, but how to grab the right directory for the imagery
    sat_res = 60 if pred_year >= 2016 else 100
    imagery_dir = paths.SCRATCH+f"naip/{pred_year}/{cfg.state}_{sat_res}cm_{pred_year}"
    rasters = naip.find_rasters_polygon(naip_shp, pred_outline.geometry.iloc[0], imagery_dir)
    # if predictions already exist, ignore the pre-predicted files
    already_done = [r.split('/')[-1].split(f'_{pred_year}')[0] for r in glob.glob(f"{save_dir}*/*_raw*.tif")]
    rasters = [r for r in rasters if r.split('/')[-1].split(f'_{pred_year}')[0] not in already_done]
    print(f"{len(already_done)} rasters completed, {len(rasters)} more to go")
    # load in climate rasters if necessary
    # TODO: make enum
    if clim_rasters == None:
        clim_rasters = build.get_bioclim_rasters(ras_name=cfg.clim_ras, timeframe=cfg.clim_time, state=cfg.state)

    # set up save directory
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if n_processes > 1:
        # set up parallel
        ras_pars = utils.partition(rasters, n_processes)
        lock = multiprocessing.Manager().Lock()
        pool =  multiprocessing.Pool(n_processes)
        res_async = [pool.apply_async(predict_rasters_parallel, args=(i, lock, ras, save_dir, cfg, device, batch_size, epoch, pred_year, pred_types, alpha_type, pred_res, sat_res, impute_climate, clim_rasters, pred_specs)) for i, ras in enumerate(ras_pars)]
        res_files = [r.get() for r in res_async]
        pool.close()
        pool.join()

    else:
        res_files = predict_rasters_serial(rasters=rasters, 
                                      parent_dir=save_dir, 
                                      model_config=cfg, 
                                      device_no=device, 
                                      batch_size=batch_size, 
                                      epoch=epoch, 
                                      pred_year=pred_year,
                                      pred_types=pred_types, 
                                      alpha_type = alpha_type,
                                      resolution=pred_res, 
                                      impute_climate=impute_climate, 
                                      sat_res=sat_res,
                                      clim_rasters=clim_rasters,
                                      pred_specs=pred_specs)

    return res_files

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument('--shape_pth', type=str, help='relative path to location of shapely file storing tiffs to predict with', required=True)
    args.add_argument('--parent_dir', type=str, help='what parent directory to save the tiffs in. Full path expansion is: {paths.RASTERS}/{parent_dir}/{file_name}', required=True) 
    args.add_argument('--pred_year', type=int, help='What year of imagery to make predictions with', default='2012')
    # https://stackoverflow.com/questions/15753701/how-can-i-pass-a-list-as-a-command-line-argument-with-argparse
    args.add_argument('--pred_types', nargs = '+', help='What type/s of predictions to make', choices = naip.Prediction.valid(), default=['RAW'])
    args.add_argument('--alpha_type', type = str, help='What type of alpha prediction to make', choices =  naip.Alpha.valid(), default='SUM')
    args.add_argument('--exp_id', type=str, help='Experiment ID for model to use for mapmaking', required=True)
    args.add_argument('--band', type=str, help='Band which model to use for mapmaking was trained on', required=True)
    # TODO: change loss and models to enums
    args.add_argument('--loss', type=str, help='Loss function used to train mapmaking model', required=True, choices=losses.Loss.valid())
    args.add_argument('--architecture', type=str, help='Architecture of mapmaking model', required=True, choices=mods.Model.valid())
    args.add_argument('-sp','--species_to_predict', nargs='+', help='Which species to save prediction maps for, in style "Genus species"', default=None)
    args.add_argument('--state', type=str, help='What state predictions are being made int', default='ca')
    args.add_argument('--epoch', type=int, help='what model epoch to use for making maps', required=True)
    args.add_argument('--batch_size', type=int, help='what size batch to use for making map inference', default=10)
    args.add_argument('--pred_resolution', type=int, help='what meter resolution to make map', default=utils.IMG_SIZE)
    args.add_argument('--sat_resolution', type=float, help='what meter resolution to up / downsample base imagery to', default=1.0)
    args.add_argument('--device', type=int, help="Which CUDA device to use. Set -1 for CPU", default=-1)
    args.add_argument('--processes', type=int, help="How many worker processes to use for mapmaking", default=1)
    args.add_argument('--impute_climate', action='store_true', help="whether to impute the climate for locations with no bioclim coverage")
    args.add_argument('--clim_ras', type=str, help='Which bioclim raster to use', default='current')
    args.add_argument('--clim_time', type=str, help='Whether to do future or current climate', default='current', choices =['current', 'future'])
    args.add_argument('--add_preds', action='store_true', help="add additional prediction types instead of making new predictions")
    args.add_argument('--overwrite', action='store_true', help="whether to overwrite existing files when calculating additional attributes")
    args, _ = args.parse_known_args()
  
    if args.processes > 1:
        multiprocessing.set_start_method('spawn')
    # load config
    cnn = {
        'exp_id': args.exp_id,
        'band' : args.band, 
        'loss': args.loss,
        'model': args.architecture
    }

    # add new kind of prediction
    if args.add_preds:
        additional_attributes_pertiff(parent_dir  = args.parent_dir,
                             pred_res = args.pred_resolution,
                             pred_year = args.pred_year,
                             band = args.band,
                             exp_id = args.exp_id,
                             epoch  = args.epoch,
                             pred_types = args.pred_types,
                             alpha_type = args.alpha_type,
                             processes = args.processes,
                             pred_specs=args.species_to_predict,
                             overwrite=args.overwrite)

    # else, just make full predictions
    else:
        cfg = run.load_config(**cnn)
        # read in polygon
        bound_shp = gpd.read_file(f"{paths.SHPFILES}{args.shape_pth}")
        predict_rasters_list(pred_outline = bound_shp,
                             pred_types = args.pred_types,
                             alpha_type = args.alpha_type,
                             parent_dir  = args.parent_dir,
                             cfg  = cfg,
                             epoch = args.epoch,
                             band = args.band,
                             state = args.state,
                             pred_year = args.pred_year,
                             device = args.device,
                             n_processes = args.processes, 
                             batch_size = args.batch_size,
                             pred_res = args.pred_resolution,
                             sat_res = args.sat_resolution,
                             impute_climate = args.impute_climate,
                             pred_specs = args.species_to_predict)