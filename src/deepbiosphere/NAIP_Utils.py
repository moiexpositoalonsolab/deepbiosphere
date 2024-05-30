# rasterio packages
import rasterio
from rasterio import merge
from rasterio import Affine
from rasterio import profiles
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling

# GIS packages
import shapely as shp
import geopandas as gpd
from pyproj import Transformer as projTransf
from shapely.geometry import Point, Polygon

# deepbio functions
import deepbiosphere.Models as mods
import deepbiosphere.Dataset as dataset
import deepbiosphere.Build_Data as build
import deepbiosphere.Run as run
from deepbiosphere.Losses import Loss
import deepbiosphere.Utils as utils
from deepbiosphere.Utils import paths

# torch / stats functions
import torch
import numpy as np
from scipy.spatial import distance
import torchvision.transforms.functional as TF

# misc functions
import os
import math
import glob
import time
import enum
import copy
from tqdm import tqdm
import multiprocessing
from enum import Enum
from functools import partial
from functools import reduce
from typing import List, Tuple
from types import SimpleNamespace

# ---------- Types ---------- #

# only needs to be simple namespace
# b/c no CLI type checking done
CRS = SimpleNamespace(
    GBIF_CRS = 'EPSG:4326',
    BIOCLIM_CRS ='EPSG:4326',
    NAIP_CRS_1 = 'EPSG:26911',
    NAIP_CRS_2 = 'EPSG:26910')
    
# legal types of predictions to make
# on a set of rasters
class Prediction(Enum, metaclass=utils.MetaEnum):
    RAW = 'raw'
    FEATS = 'features'
    PER_SPEC = 'per_species'
    ALPHA = 'alpha'
    BETA = 'beta'
    
    
# ---------- alpha diversity function types ---------- #


def alpha_thres(pred : torch.tensor, threshold=0.8):
    pred = pred >= threshold # binary threshold
    return pred.sum(axis=0) # sum across all species
    
def alpha_sum(pred : torch.tensor):
    # sum probability across all species
    return pred.sum(axis=0) 


# ---------- Beta diversity convolution function types ---------- #

# assumes that wind is [nspec, filt, filt]
def combined(wind, dist_fn, agg_fn):
    cx, cy = wind.shape[1]//2, wind.shape[2]//2
    center = wind[:,cx,cy]
    diffs = []
    for i in range(wind.shape[1]):
        for j in range(wind.shape[2]):
            if i == cx and j == cy:
                continue
            diffs.append(dist_fn(wind[:,i,j], center))
    return agg_fn(diffs)

sx = np.array([
    [1,0,-1],
    [2,0,-2],
    [1,0,-1]
])
sy = np.array([
    [-1,-2,-1],
    [0,0,0],
    [1,2,1]
])

def sobel(wind, dist_fn, agg_fn):
    # convolve wind and sobel filters
    gx = np.multiply(wind, sx).sum()
    gy = np.multiply(wind, sy).sum()
    return agg_fn([gx, gy]), math.atan2(gx, gy)

px = np.array([
    [-1,0,1],
    [-1,0,1],
    [-1,0,1]
])
py = np.array([
    [-1,-1,-1],
    [0,0,0],
    [1,1,1]
])
def prewitt(wind, dist_fn, agg_fn):
    gx = np.multiply(wind, px).sum()
    gy = np.multiply(wind, py).sum()
    return agg_fn([gx, gy]), math.atan2(gx, gy)

cx = np.array([.5, 0. -.5])
def central(wind, dist_fn, agg_fn):
    gx = np.dot(cx, wind).sum()
    gy = np.dot(cx.T, wind).sum()
    return agg_fn([gx,gy]), math.atan2(gx, gy)

lx = np.array([
    [0,1,0],
    [1,-4,1],
    [0,1,0]
])
ly = np.array([
    [-1,-1,-1],
    [-1,8,-1],
    [-1,-1,-1]
])
def laplace(wind, dist_fn, agg_fn):
    gx = np.multiply(wind, lx).sum()
    gy = np.multiply(wind, ly).sum()
    return agg_fn([gx, gy]), math.atan2(gx, gy)


# ---------- Function enum types ---------- #

# legal functions to use
# when calculating alpha diversity
# partial() wrapper is necessary so both dot and bracket notation
# work with function calls, ie: DistanceFns.L2 and DistanceFns['L2']
# https://stackoverflow.com/questions/40338652/how-to-define-enum-values-that-are-functions
class Alpha(utils.FuncEnum, metaclass=utils.MetaEnum):
    THRES = partial(alpha_thres)
    SUM   = partial(alpha_sum)

# legal distance functions to use
# when calculating beta diversity
# or community change
# partial() wrapper is necessary so both dot and bracket notation
# work with function calls, ie: DistanceFns.L2 and DistanceFns['L2']
# https://stackoverflow.com/questions/40338652/how-to-define-enum-values-that-are-functions
class Distance(utils.FuncEnum, metaclass=utils.MetaEnum):
    L2       = partial(lambda x, y: (np.linalg.norm(x-y)))
    COSINE   = partial(lambda x, y: (distance.cosine(x,y)))
    DOT_PROD = partial(lambda x, y: (np.dot(x,y)))
    KL_DIV   = partial(lambda x, y: (sum([xx*np.log(xx/yy) for xx,yy in zip(x,y)])))

# legal filters to use for beta
# diversity calculation
class Filter(utils.FuncEnum, metaclass=utils.MetaEnum):
    SOBEL    = partial(sobel)
    PREWITT  = partial(prewitt)
    CENTRAL  = partial(central)
    COMBINED = partial(combined)
    LAPLACE  = partial(laplace)
    
# legal ways to aggregate beta predictions
# across species
class Aggregation(utils.FuncEnum, metaclass=utils.MetaEnum):
    # same as np.linalg.norm(x, 1)
    SUM  = partial(lambda x: sum([abs(y) for y in x]))
    # same as np.linalg.norm(x)
    NORM = partial(lambda x: math.sqrt(sum([y**2 for y in x])))
    # same as np.linalg.norm(x, 1)/ len(x)
    AVG = partial(lambda x: sum([abs(y) for y in x]) / len(x))
    MULT = partial(lambda z: reduce(lambda x, y: x*y, z))


# ---------- alpha diversity calculation functions ---------- #
    
def predict_alpha(pred : np.array,
                  alpha_func : str = 'SUM'):

    # type check + get function
    alphn = Alpha[alpha_func]
    return alphn(pred)

    
# ---------- beta diversity calculation functions ---------- #

# user's job to ensure filter function and perspecies are compatible
# convolves network predictions to determine how different the local neighborhood of predictions is
# preds: the predictions from the network, last 2 dimensions should be the height and width of your predictions
# filt_size: the dimensions of the filter to pass over the predictions (width, height) tuple
# dist_name: key for dist_fns dict, determines what distance function to use to calculate difference between predictions
# filter: one of the keys for the filter function, determines which convolutional filter to use
# agg_name: one of the keys from the above aggregate_fns, determines which aggregation strategy to use
# Remember, the first axis must be the species axis!
def convolve(probas : torch.tensor, 
             filt_size : Tuple[int,int] = (3,3), 
             dist_name : str = 'L2', 
             fil_name : str = 'COMBINED', 
             agg_name : str = 'NORM', 
             nodata=np.nan):

    # type check + get function
    agg_fn = Aggregation[agg_name]
    dist_fn = Distance[dist_name]
    filter_fn = Filter[fil_name]
    
    height = probas.shape[1]
    width = probas.shape[2]
    # generate results array. Will lose a few 
    # pixels on the end due to convolution
    # -1 is to ensure last valid pixel captured
    rowbuffer, colbuffer = filt_size[0]-1, filt_size[1]-1
    nheight, nwidth = height-rowbuffer, width-colbuffer
    convo = np.full([nheight,nwidth], nodata, dtype=np.float64)
    # loop through
    for i in range(nheight): 
        for j in range(nwidth): 
            # grab all neighboring pixels
            wind = probas[:, i:i+filt_size[0], j:j+filt_size[1]]
            # and calculate neighborhood distance 
            convo[i,j] = filter_fn(wind, dist_fn, agg_fn)
    return convo

# wrap convolve with rasterio handling    
def predict_beta(preds, transform):
    if not torch.is_tensor(preds):
        pred = torch.tensor(preds)
    # default convolve parameters are those I
    #  found produced the most accurate maps.
    beta_pred = convolve(preds)

    # with a 3x3 convolution, lose outside row of pixels
    # so update transform to match new boundaries
    height, width = preds.shape[1], preds.shape[2]
    # last row/col is one less than before from pixel loss
    nheight, nwidth = height-1, width-1 # 1 changes if filt-size does
    hig, wid = beta_pred.shape[0], beta_pred.shape[1]
    west,  north = rasterio.transform.xy(transform, [1],[1])
    east, south = rasterio.transform.xy(transform, [nheight],[nwidth])
    transf = rasterio.transform.from_bounds(west[0], 
                                            south[0], 
                                            east[0], 
                                            north[0], 
                                            wid, hig)
    # get true resolution as well
    # resolution order according to rasterio
    # is (width, height)
    out_res = (width/wid, height/hig)
    bounds = rasterio.transform.array_bounds(hig, wid, transf)
    return beta_pred, transf, out_res, bounds

# ---------- geopandas helper functions ---------- ##

def get_state_outline(state, file=f"{paths.SHPFILES}gadm36_USA/gadm36_USA_1.shp"):
    # get outline of us
    us1 = gpd.read_file(file) # the state's shapefiles
    stcol = [h.split('.')[-1].lower() for h in us1.HASC_1]
    us1['state_col'] = stcol
    # only going to use California
    shps = us1[us1.state_col == state]
    return shps

def load_naip_bounds(base_dir : str, state: str, year : str):
    return gpd.read_file(glob.glob(f"{base_dir}/naip_tiffs/{state}_shpfl_{year}/*.shp")[0])

def find_rasters_point(gdf, point,  base_dir  : str):
    rasters = gdf[gdf.contains(point)]
    rasters = [f"{base_dir}/{fman.APFONAME[:5]}/{'_'.join(fman.FileName.split('_')[:-1])}.tif" for _, fman in rasters.iterrows()]
    return rasters

def find_rasters_polygon(gdf, polygon, base_dir):

    rasters = gdf[gdf.intersects(polygon)]
    rasters = [f"{base_dir}/{fman.APFONAME[:5]}/{'_'.join(fman.FileName.split('_')[:-1])}.tif" for _, fman in rasters.iterrows()]
    return rasters

def find_rasters_gdf(raster_gdf, slice_gdf, base_dir):
    raster_gdf = raster_gdf.to_crs(slice_gdf.crs)
    select = np.zeros((len(raster_gdf)))
    for shape in slice_gdf.geometry:
        select[raster_gdf.intersects(shape).values] = True
    subset = raster_gdf[select > 0]
    return [f"{base_dir}/{fman.APFONAME[:5]}/{'_'.join(fman.FileName.split('_')[:-1])}.tif" for _, fman in subset.iterrows()], subset

def bounding_box_to_polygon(bounds):
    return shp.geometry.Polygon([(bounds.left, bounds.top), (bounds.left, bounds.bottom), (bounds.right, bounds.bottom), (bounds.right, bounds.top)])

def get_window(transform, xs, ys):
    rs, cl = rasterio.transform.rowcol(transform, xs, ys, op=round) # may need to change op..
    col_off, row_off = min(cl), min(rs)
    width = max(cl)-min(cl)
    height = max(rs) - min(rs)
    return rasterio.windows.Window(col_off, row_off, width, height)


# ---------- handling climate ---------- #

def get_climate(bioclim_rasters, bioclim_transf, bioclim_crs, affine, curr_col, curr_row, crs, impute_clim):
    # grab the x / y of top left corner of each image for the climate
    xys = rasterio.transform.xy(affine, *zip(*[(c,curr_col) for c in curr_row]))
    # transform x,y into lat/lon crs
    transform = projTransf.from_crs(crs, bioclim_crs)
    xys = list(transform.itransform(zip(*xys)))
    # and grab the climate from the raster
    clims = np.full([len(xys), len(bioclim_rasters)], 0.0)
    nan_masks = np.full([len(xys), len(bioclim_rasters)], False)
    for i, (y, x) in enumerate(xys):
        row, col = rasterio.transform.rowcol(bioclim_transf, x, y)

        # if out of bounds of climate, return nan
        if (row >= bioclim_rasters.shape[1]) or (col >= bioclim_rasters.shape[2]) or (row < 0) or (col < 0):
            if impute_clim:
                width, height = bioclim_rasters.shape[2], bioclim_rasters.shape[1]
                # find and impute nearest 
                # unmasked climate pixel
                res = impute_climate(bioclim_rasters, np.nan, 1, row,col, width, height)
            else:
                # mask out missing values and 
                # convert to nan post-inference
                # with return mask
                nan_masks[i, :] = True
                continue
        # assumes climate rasters are aligned!
        res = bioclim_rasters[:,row,col]
        # check if masked / nan (both to be safe)
        if (np.ma.is_masked(res[0]) or np.isnan(res[0])):
            if impute_clim:
                width, height = bioclim_rasters.shape[2], bioclim_rasters.shape[1]
                # find and impute nearest 
                # unmasked climate pixel
                res = impute_climate(bioclim_rasters, np.nan, 1, row,col, width, height)
            else:
                # mask out missing values and 
                # convert to nan post-inference
                # with return mask
                nan_masks[i, :] = True
                
        clims[i,:] = res
        
    return clims, nan_masks

# recursive function to iteratively search
# for next-nearest non-masked climate pixel
def impute_climate(arr, res, curr_diff, row, col, width, height):
    # edge case when the whole raster is nans
    # just break and return in that case
    if (curr_diff > width) or (curr_diff > height):
        return res
    # make a k-dimension box around pixel
    for i in range(max(row-curr_diff,0), min(row+curr_diff+1, height)):
        for j in range(max(col-curr_diff, 0), min(col+curr_diff+1, width)):
            # look at current box pixel
            res = arr[:,i,j]
            # if not masked or nan, keep it
            if (not np.isnan(res[0])) and (not np.ma.is_masked(res[0])):
                return res
    # if no dice, increase the neighborhood
    #size and check next-largest box
    return impute_climate(arr, res, curr_diff+1, row, col, width, height)

    
# ---------- raw prediction calculation using CNN model ---------- #

def predict_model(dat, 
            affine,
            crs,
            model,
            batch_size, 
            device, 
            img_size, 
            n_specs, 
            res, 
            means, 
            std, 
            save_name,
            use_climate=True,
            imclim=False,
            bioclim_rasters=None,
            bioclim_transf=None,
            bioclim_crs=None,
            disable_tqdm=False,
            pred_specs=None,
            spec_names=None):

    

    dwidth, dheight = dat.shape[2], dat.shape[1]
    # generate starting indices for data convolved to new resolution
    i_ind = list(range(0, dwidth, res))
    j_ind = list(range(0, dheight, res))
    # remove all chunks that aren't large enough
    while (dwidth - i_ind[-1]) < img_size:
        i_ind.pop()
    while (dheight - j_ind[-1]) < img_size:
        j_ind.pop()

    nwidth, nheight = i_ind[-1]+img_size, j_ind[-1]+img_size
    wid, hig = len(i_ind), len(j_ind)
    # make sure batching won't fall off image
    assert batch_size < hig, f"Batch ({batch_size}) should be < height {hig} (width: {wid})"
    # make receiver array
    # dtype=np.float32 For now will try and see if default 
    # float64 works. Some versions of GDAL only can work 
    # with float32 but we'll see
    result = np.full([n_specs, hig, wid],np.nan)
    if pred_specs is not None:
        mapping = {s: i for s, i in zip(spec_names, range(len(spec_names)))}
        idxs = [mapping[i] for i in pred_specs] 
    else:
        idxs = range(n_specs)
    # actually predict
    with torch.no_grad():
        if not disable_tqdm:
            prog = tqdm(total=wid*math.ceil(hig/batch_size), unit="batch", desc=f'{save_name} species prediction ({wid}x{hig})')
        # go column by column across raster
        for i, curr_col in enumerate(i_ind):

            # chunks up the column into batches
            # chunks returns a jagged array but
            # that's okay np handles it below
            chunked_rows = utils.chunks(j_ind, batch_size)
            # also batch up start idxs in result array
            batched_rows = range(0, hig, batch_size)
            for j, curr_row in zip(batched_rows, chunked_rows):
                
                # grab each image at the resolution we prefer from image raster
                imgs = [dat[:, c:c+img_size, curr_col:curr_col+img_size] for c in curr_row]
                imgs = np.stack(imgs)
                # normalize, scale
                imgs = utils.scale(imgs, out_range=(0,1), min_=0, max_=255)
                imgs = TF.normalize(torch.tensor(imgs, dtype=torch.float), means, std)
                imgs = imgs.to(device)
                # add climate if using it
                # might be slightly faster to
                # move this if check toanother
                # method but this saves boilerplate
                # so keeping for now
                if use_climate:
                    
                    clim, nans = get_climate(bioclim_rasters, bioclim_transf, bioclim_crs, affine, curr_col, curr_row, crs, imclim)
                    clim = torch.tensor(clim, dtype=torch.float)
                    clim = clim.to(device)
                    inputs = (imgs, clim)
                else:
                    inputs = imgs
                    
                out = model(inputs)
                # ignore genus and famliy prediction
                # for deepbiosphere models
                if isinstance(out, tuple):
                    # only take species prediction
                    out = np.squeeze(out[0].detach().cpu().numpy())
                else:
                    out = np.squeeze(out.detach().cpu().numpy())
                
                # if using climate and masking missing climate, mask out missing
                if use_climate:
                    nans = nans.sum(axis=1)
                    # save a touch of time by summing across axes only once
                    if nans.sum() > 0:
                        mask = nans > 0
                        out[mask, :] = np.nan
                # flip outputs to move species axis to 1st axis,
                # batch axis to row
                out = out.T
                # filter to only species we want to keep
                out = out[idxs,:]
                result[:,j:j+batch_size,i] = out    
                if not disable_tqdm:
                    prog.update(1)
        
        if not disable_tqdm:
            prog.close()
    
    west,  north = rasterio.transform.xy(affine, [0],[0])
    east, south = rasterio.transform.xy(affine, [nheight],[nwidth])
    transf = rasterio.transform.from_bounds(west[0], south[0], east[0], north[0], wid, hig)
    # get true resolution as well
    out_res = (nwidth/wid, nheight/hig)
    # finally, get new bounds outline of image
    bounds = rasterio.transform.array_bounds(hig, wid, transf)
    return result, transf, out_res, bounds



# ---------- raw prediction calculation using CNN model ---------- #

def predict_raster(raster,
                save_dir : str,
                save_name : str,
                model, # could be multiple types of models
                model_config : SimpleNamespace,
                device  : int,
                batch_size : int,
                pred_types : List[str] = ['RAW'], # list of legal prediction types
                alpha_type : str = 'SUM',
                resolution : int = utils.IMG_SIZE,
                img_size : int = utils.IMG_SIZE, # TODO: change this to any number above 20 for standard convolutional approach
                impute_climate: bool = True, 
                clim_rasters : List = None,
                disable_tqdm : bool = False,
                sat_res : int = None,
                pred_specs : List[str] = None): # resolution of satellite imagery for prediction, in meters

    # typecheck prediction parameters
    pred_types = [Prediction[pred_type] for pred_type in pred_types]
    alpha_type = Alpha[alpha_type]
    # load in necessary metadata
    dset_metadata = dataset.load_metadata(model_config.dataset_name)
    # adding a new element to config which saves the training means used
    # f"naip_{model_config.year}" 
    means = dset_metadata.dataset_means[model_config.image_stats]['means']
    stds = dset_metadata.dataset_means[model_config.image_stats]['stds']
    # if resolution < 20:
    #     raise ValueError("resolution needs to be at least 20 for model to work!")

    spec_names = dataset.get_specnames(dset_metadata)
    n_specs = model_config.nspecs
    
    use_climate = dataset.DataType[model_config.datatype] == dataset.DataType.JOINT_NAIP_BIOCLIM
    if (use_climate and (clim_rasters is None)):
        clim_rasters = build.get_bioclim_rasters(state=model_config.state)

    bioclim_ras = np.vstack([r[0] for r in clim_rasters]) if use_climate else None
    bioclim_transf = clim_rasters[0][1] if use_climate else None
    bioclim_crs = clim_rasters[0][3] if use_climate else None
    # check if it's a file, read in if so
    if isinstance(raster, rasterio.io.DatasetReader):
        assert raster.res[0] <= resolution, "resolution of prediction must be larger than image res!"
        affine = raster.transform
        crs = raster.crs
        if sat_res is not None:
            # convert to cm with /100
            raster, affine = merge.merge([raster], res=(sat_res/100, sat_res/100))
        else:
            raster = raster.read()
    # else if raster is already read out raster + tuple
    # then expand out those parameters
    elif isinstance(raster, tuple):
        raster, affine, crs = raster
        assert len(raster.shape) == 3, 'wrong dimensions for raster!'
        assert raster.shape[0] < raster.shape[1], f"first dimension should be bands! Shape is: {raster.shape}"
    elif (isinstance(raster, str)) and (os.path.isfile(raster)):
        raster = rasterio.open(raster)
        affine = raster.transform
        crs = raster.crs
        if sat_res is not None:
            raster, affine = merge.merge([raster], res=(sat_res, sat_res))
        else:
            raster = raster.read()
    else:
        raise ValueError(f"raster is invalid datatype! {raster}")
        
    # otherwise read it from the open tiff
    # now get the actual predictions
    pred, pred_aff, out_res, out_bounds = predict_model(dat=raster, 
                                                affine=affine, 
                                                crs=crs,
                                                model=model,
                                                batch_size=batch_size, 
                                                device=device, 
                                                img_size=img_size, 
                                                n_specs=n_specs if pred_specs is None else len(pred_specs), 
                                                res=resolution, 
                                                means=means, 
                                                std=stds, 
                                                save_name=save_name,
                                                use_climate=use_climate,
                                                imclim=impute_climate,
                                                bioclim_rasters=bioclim_ras,
                                                bioclim_transf=bioclim_transf,
                                                bioclim_crs=bioclim_crs,
                                                disable_tqdm=disable_tqdm,
                                                pred_specs=pred_specs,
                                                spec_names=spec_names if pred_specs is None else pred_specs)
  
    files = []
    # convert to probabilities
    if not torch.is_tensor(pred):
        pred = torch.tensor(pred)
    # softmax or sigmoid depending on model loss
    loss_type = Loss[model_config.loss]
    if loss_type in [Loss.WEIGHTED_CE, Loss.CE]:
        print("softmaxing predictions")
        pred = torch.softmax(pred, axis=0).numpy()
    else:
        pred = torch.sigmoid(pred).numpy()
    
    # and save out predictions
    for pred_type in pred_types:
        # save species predictions
        if pred_type is Prediction.PER_SPEC:
            files += save_tiff_per_species(save_dir=save_dir,
                                        save_name=save_name,
                                        loss_type=model_config.loss,
                                        preds=pred,
                                        transf=pred_aff,
                                        crs=crs,
                                        null_val=np.nan,
                                        out_res=out_res,
                                        bounds=out_bounds,
                                        spec_names=spec_names,
                                        pred_specs=pred_specs)

            
        elif pred_type is Prediction.RAW:
            files.append(save_tiff(save_dir=save_dir,
                                save_name=save_name,
                                pred_type=Prediction.RAW.value,
                                preds=pred,
                                transf=pred_aff,
                                crs=crs,
                                null_val=np.nan,
                                out_res=out_res,
                                bounds=out_bounds,
                                band_names=spec_names if pred_specs is None else pred_specs))
        elif pred_type is Prediction.FEATS:
            model.encode = True
            num_feats = [f"{s}" for s in list(range(model_config.feature_extraction_dim))]
            n_feats = model_config.feature_extraction_dim # not species, but features
            feats, feat_aff, feat_out_res, feat_out_bounds = predict_model(dat=raster, 
                                    affine=affine, 
                                    crs=crs,
                                    model=model,
                                    batch_size=batch_size, 
                                    device=device, 
                                    img_size=img_size, 
                                    n_specs=n_feats, 
                                    res=resolution, 
                                    means=means, 
                                    std=stds, 
                                    save_name=save_name,
                                    use_climate=use_climate,
                                    imclim=impute_climate,
                                    bioclim_rasters=bioclim_ras,
                                    bioclim_transf=bioclim_transf,
                                    bioclim_crs=bioclim_crs,
                                    disable_tqdm=disable_tqdm)
            files.append(save_tiff(save_dir=save_dir,
                                save_name=save_name,
                                pred_type=Prediction.FEATS.value,
                                preds=feats,
                                transf=feat_aff,
                                crs=crs,
                                null_val=np.nan,
                                out_res=feat_out_res,
                                bounds=feat_out_bounds,
                                band_names=num_feats))
            model.encode = False
        # Generate alpha diversity from species predictions
        elif pred_type is Prediction.ALPHA:
            # prevent modification of
            # underlying pred array
            a_pred = np.copy(pred)
            alpha_pred = predict_alpha(a_pred, alpha_type)
            alpha_pred = np.expand_dims(alpha_pred, axis=0)
            files.append(save_tiff(
                save_dir=save_dir,
                save_name=save_name,
                pred_type=Prediction.ALPHA.value,
                preds=alpha_pred,
                transf=pred_aff,
                crs=crs,
                null_val=-1 if alpha_type is Alpha.THRES else np.nan,
                out_res=out_res,
                bounds=out_bounds,
                band_names=[Prediction.ALPHA.value]))

        # Generate beta diversity from species predictions
        elif pred_type is Prediction.BETA:
            # prevent modification of
            # underlying pred array
            b_pred = np.copy(pred)
            beta_pred, beta_aff, beta_res, beta_bounds = predict_beta(b_pred, pred_aff)
            beta_pred= np.expand_dims(beta_pred, axis=0)
            # and save out
            files.append(save_tiff(
                save_dir=save_dir,
                save_name=save_name,
                pred_type=Prediction.BETA.value,
                preds=beta_pred,
                transf=beta_aff,
                crs=crs,
                null_val=np.nan,
                out_res=beta_res,
                bounds=beta_bounds,
                band_names=[Prediction.BETA.value]))


    return files

# ---------- raster file saving utils ---------- #
    
def get_specs_from_files(files):
    # old, if file is per_species
    # file = files[0]
    # prepath = file.rsplit('/', 1)[0]
    # matching = [f for f in files if prepath in f]
    # # grab only species names
    # return ['_'.join(m.split('_')[-2:]).split('.tif')[0] for m in matching]
    if isinstance(files[0], str):
        ras1 = rasterio.open(files[0])
        ras2 = rasterio.open(files[-1])
    else: 
        ras1 = files[0]
        ras2 = files[-1]
    specs = ras1.descriptions
    if specs == ras2.descriptions:
        return specs
    else:
        raise ValueError('band descriptions do not match across files!')
        
def save_tiff_per_species(save_dir,
                          save_name,
                          preds,
                          transf,
                          crs,
                          null_val,
                          out_res,
                          bounds,
                          spec_names,
                         pred_specs=None,
                         overwrite=False):

    # make directory for each tif
    new_dir = f"{save_dir}{save_name}/"
    
    # make subdir if it doesn't exist
    # if the parent directory doesn't 
    # exist, will fail to prevent creating
    # a directory in an unwanted place
    if not os.path.isdir(new_dir):
        os.mkdir(new_dir)
    files = []
    if pred_specs is None:
       # save out each species to file
        for pred, spec in zip(preds, spec_names):
            if not overwrite:
                if check_file_exists(new_dir, 'probability', spec.replace(' ', '_')):
                    continue
            # massage into correct dimensions for rasterio
            pred = np.expand_dims(pred, axis=0)
            files.append(save_tiff(
                save_dir=new_dir,
                save_name='probability',
                pred_type=spec.replace(' ', '_'),
                preds=pred,
                transf=transf,
                crs=crs,
                null_val=np.nan,
                out_res=out_res,
                bounds=bounds,
                band_names=[spec.replace(' ', '_')]))
        

    else:
        mapping = {s: i for s, i in zip(spec_names, range(len(spec_names)))}
        assert len(mapping) == preds.shape[0]
        for spec in pred_specs:
            if not overwrite:
                if check_file_exists(new_dir, 'probability', spec.replace(' ', '_')):
                    continue
            specidx = mapping[spec]
            pred = preds[specidx,:,:]
            # massage into correct dimensions for rasterio
            pred = np.expand_dims(pred, axis=0)
            files.append(save_tiff(
                save_dir=new_dir,
                save_name='probability',
                pred_type=spec.replace(' ', '_'),
                preds=pred,
                transf=transf,
                crs=crs,
                null_val=np.nan,
                out_res=out_res,
                bounds=bounds,
                band_names=[spec.replace(' ', '_')]))    
    return files
    

def check_file_exists(save_dir, save_name, pred_type):
    # same as save_tiff
    fname = f"{save_dir}{save_name}_{pred_type}.tif"
    return os.path.isfile(fname)

    
def save_tiff(save_dir,
              save_name,
              pred_type,
              preds, 
              transf, 
              crs, 
              null_val, 
              out_res, 
              bounds,
              band_names):
    
    assert len(preds.shape) > 2, f"need to add bands axis! {preds.shape}"
    
    width = preds.shape[2]
    height =  preds.shape[1]
    
    out_profile = rasterio.profiles.DefaultGTiffProfile()
    out_profile['res'] = out_res
    out_profile['bounds'] = bounds
    out_profile['transform'] = transf
    out_profile['crs'] = crs
    out_profile['height'] = height
    out_profile['width'] = width
    out_profile['count'] = preds.shape[0]
    out_profile['nodata'] = null_val
    out_profile['dtype'] = preds.dtype
    # recs taken from naip tiff profile
    out_profile['interleave'] = 'pixel'
    out_profile['compress'] = 'deflate'
    out_profile['blockxsize'] =512
    out_profile['blockysize'] =512
    out_profile['BIGTIFF'] = 'IF_SAFER'
    fname = f"{save_dir}{save_name}_{pred_type}.tif"
    with rasterio.open(fname, 'w', **out_profile) as dst:
        dst.write(preds, range(1,len(band_names)+1), masked=False)
        dst.descriptions = band_names
    return fname



def merge_lots_of_tiffs(parent_dir, save_dir, pred_type, alpha_type='SUM', files=None, filelimit=1000, crs=CRS.NAIP_CRS_2,band_names=None,resolution=None, null_val=np.nan, spec=None):

    
    if files is None:
        files = glob.glob(f"{paths.RASTERS}{parent_dir}/{save_dir}/*/*{pred_type}.tif")
        files = [f for f in files if ('fully_merged' not in f) or ('merging_temp' not in f)]

        
    print(f"merging {len(files)} files")
    pred_type = Prediction[pred_type]
    alpha_type = Alpha[alpha_type]
    chunked_files = [files[i:i+filelimit] for i in range(0, len(files), filelimit)]
    merged = [] 
    for i, chunk in tqdm(enumerate(chunked_files), desc='merging chunks', unit=' tiff', total=len(chunked_files)):
        # read in files
        files = [rasterio.open(f) for f in chunk]
        warped_files = [rasterio.vrt.WarpedVRT(f, crs=crs) for f in files]
        # merge warped files
        if resolution is None:
            preds, aff = rasterio.merge.merge(warped_files)
        else:
            preds, aff = rasterio.merge.merge(warped_files, res=resolution)
        # write merged predictions out to file
        height = preds.shape[1]
        width = preds.shape[2]
        bounds = rasterio.transform.array_bounds(height, width, aff)
        new_dir = f"{paths.RASTERS}{parent_dir}/{save_dir}/merging_temp/"
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
        if not os.path.isdir(new_dir):
            os.mkdir(new_dir)
        merged.append(save_tiff(
            save_dir=new_dir,
            save_name=save_dir,
            pred_type=f"{pred_type.value}_{i}",
            preds=preds,
            transf=aff,
            crs=crs,
            null_val= null_val,
            out_res=aff[0],
            bounds=bounds,
            band_names=[pred_type.value] if band_names is None else band_names))
        _ = [f.close() for f in warped_files]
        _ = [f.close() for f in files]
    meta_warp = [rasterio.open(f) for f in merged]
    dest_pth = f"{paths.RASTERS}{parent_dir}/{save_dir}/fully_merged/"
    if not os.path.isdir(dest_pth):
        os.mkdir(dest_pth)
    dst_file = f"{dest_pth}{save_dir}_{pred_type.value}_merged.tif" if spec is None else f"{dest_pth}{save_dir}_{spec}_merged.tif"
    rasterio.merge.merge(meta_warp, dst_path=dst_file)
    return dst_file
    
    
