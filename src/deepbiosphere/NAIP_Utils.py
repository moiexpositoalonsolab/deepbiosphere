# rasterio packages
import rasterio
from rasterio import Affine
from rasterio import profiles
from rasterio.warp import calculate_default_transform, reproject
from rasterio.enums import Resampling
from rasterio.enums import Resampling

# GIS packages
import shapely as shp
import geopandas as gpd
from pyproj import Transformer as projTransf
from shapely.geometry import Point, Polygon

# deepbio functions
import deepbiosphere.Models as mods
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
from tqdm import tqdm
from functools import reduce

# ---------- CRS used ---------- ##

# standard WSG84 CRS. This is what
# CRS the GBIF observations come in
GBIF_CRS='EPSG:4326'
# The two CRS for the NAIP tifs
NAIP_CRS_1 = 'EPSG:26911'
NAIP_CRS_2 = 'EPSG:26910'
# random value to fill in for missing alpha diversity values
ALPHA_NODATA = 9999

## ---------- class type hints ---------- ##
# TOOD: remove??
#NAIP_shpfile  = gpd.geodataframe.GeoDataFrame
#Point = shapely.geometry.Point



def load_naip_bounds(base_dir : str, state: str, year : str):
    return gpd.read_file(glob.glob(f"{base_dir}/naip_tiffs/{state}_shpfl_{year}/*.shp")[0])

# use gpd.sjoin instead of find-rasters-polygon, works for finding points too, although
# contains will get the job done as well (or something else, need to check jpyntbk) TODO
# TODO: solve the whole merge bug problem thing
# mask returns an array and a transform, can just use that with rasterio.plot.show()

def find_rasters_point(gdf, point,  base_dir  : str):
    #check_bands(bands)
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

# TODO: convert to use with new code
def predict_raster_list(device_no, tiffs, cfg, res, year, means, model_pth, cfg_pth, base_dir, warp):
    # necessary to work with parallel
    if device_no == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device_no}")
        torch.cuda.set_device(device_no)
        # TODO: move the config thing
    params = config.Run_Params(basedir, cfg_path=cfg_pth)
    daset = setup_dataset(params.params.observation, basedir, params.params.organism, params.params.region, params.params.normalize, params.params.no_altitude, params.params.dataset, params.params.threshold, -1, inc_latlon=False, pretrained_dset='old_tresnet')
    # just load in model directly
    state = torch.load(basedir + model_pth, map_location=device)
    # and get size to set up new model from state
    gen = state['model_state_dict']['gen.weight']
    fam = state['model_state_dict']['fam.weight']
    spec = state['model_state_dict']['spec.weight']
    num_spec = spec.shape[0]
    num_gen = gen.shape[0]
    num_fam = fam.shape[0]
    # now actually set up model
    model = mods.TResNet_M(params.params.pretrained, num_spec, num_gen, num_fam, basedir)
    model.load_state_dict(state['model_state_dict'], strict=True)
    model = model.to(device)
    model.eval();
    spec_names = tuple(daset.inv_spec.values())
    # figure out batch size
    batchsize = batchsized(device, tiffs[0], model,params.params.batch_size, res, num_spec) # params.params.batch_size
    for raster in tqdm(tiffs):
        file = predict_raster(raster, model, batchsize, res, year, base_dir, modelname, num_spec, device, spec_names, warp, means)

def get_state_outline(state, file=f"{paths.SHPFILES}gadm36_USA/gadm36_USA_1.shp"):
    # get outline of us
    us1 = gpd.read_file(file) # the state's shapefiles
    stcol = [h.split('.')[-1].lower() for h in us1.HASC_1]
    us1['state_col'] = stcol
    # only going to use California
    shps = us1[us1.state_col == state]
    return shps

def diversity_raster_list(rasters, div, year, base_dir, modelname, warp, nodata):
    for ras in tqdm(rasters):
        diversity_raster(ras, metric=div, year=year, base_dir=base_dir, modelname=modelname, warp=warp, nodata=nodata)


def alpha_div(predictions, threshold=0.5, dtype=np.uint16):
    pred = torch.tensor(predictions)
    pred = torch.sigmoid(pred) # transform into predictions
    pred = pred >= threshold # binary threshold
    pred = pred.sum(dim=0) # sum across all species
    return pred.numpy().astype(dtype)

# convolution functions for calculating species turnover (beta diversity)

# assumes that wind is [filt, filt, num_species]
def combined(wind, dist_fn, agg_fn):
    cx, cy = wind.shape[0]//2, wind.shape[1]//2
    center = wind[cx,cy,:]
    diffs = []
    for i in range(wind.shape[0]):
        for j in range(wind.shape[1]):
            if i == cx and j == cy:
                continue
            diffs.append(dist_fn(wind[i,j,:], center))
    return agg_fn(diffs)

# assumes that wind is [filt, filt]
def combined_perspec(wind, dist_fn, agg_fn):
    cx, cy = wind.shape[0]//2, wind.shape[1]//2
    center = wind[cx,cy]
    diffs = []
    for i in range(wind.shape[0]):
        for j in range(wind.shape[1]):
            if i == cx and j == cy:
                continue
            diffs.append(dist_fn(wind[i,j], center))
    return agg_fn(diffs), np.nan
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
#     print(gx, gy)
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
dist_fns = {
    'L2' : lambda x, y: (np.linalg.norm((x-y))), # TODO: check - DONE
    'cosine' : lambda x, y: (distance.cosine(x,y)),
    'dot_prod' : lambda x, y: (np.dot(x,y)),
    'kl_div' : lambda x, y: (sum([xx*np.log(xx/yy) for xx,yy in zip(x,y)])),
    'none' : None
}
filters = {
    'sobel': sobel,
    'prewitt': prewitt,
    'central' : central,
    'combined': combined, #TODO: check
    'combined_perspec': combined_perspec,
    'laplace' : laplace
}
aggregate_fns = {
    'sum' : (lambda x: sum([abs(y) for y in x])),
    'norm' : (lambda x: math.sqrt(sum([y**2 for y in x]))), # TODO: Check DONE
    'average' : (lambda x: sum([abs(y) for y in x]) / len(x)),
    'mult' : (lambda z: reduce(lambda x, y: x*y, z))
}


# user's job to ensure filter function and perspecies are compatible
# convolves network predictions to determine how different the local neighborhood of predictions is
# preds: the predictions from the network, last 2 dimensions should be the height and width of your predictions
# filt_size: the dimensions of the filter to pass over the predictions (width, height) tuple
# sig: whether to sigmoid the network's predictions and convert to a probability vector. Generally recommended since this normalizes the values to a more reasonable range
# per_species: whether to calculate the distance per-species within nieghborhood then aggregate, or calculate distance by vector. Warning: per-species is very slow!
# dist_name: key for dist_fns dict, determines what distance function to use to calculate difference between predictions
# filter: one of the keys for the filter function, determines which convolutional filter to use
# agg_name: one of the keys from the above aggregate_fns, determines which aggregation strategy to use
# angle: whether to calculate the aggregated angle between neighboring predictions
# Remember, the first axis must be the species axis!
def convolve(probas, filt_size, sig, per_species, dist_name, fil_name, agg_name, angle=True, nodata=np.nan):

    agg_fn = aggregate_fns[agg_name]
    dist_fn = dist_fns[dist_name]
    filter = filters[fil_name]
    height = probas.shape[0]
    width = probas.shape[1]
    convo = np.full([(height-(filt_size[0]-1)),(width-(filt_size[1]-1))], nodata, dtype=np.float64)
#     convo = np.full([(height-3),(width-3)], np.nan)
    print(convo.shape, probas.shape)
    if angle:
        angles = np.full([(height-(filt_size[0]-1)),(width-(filt_size[1]-1))], nodata, dtype=np.float64)
    for i in range(convo.shape[0]):

        for j in range(convo.shape[1]): # range goes to 1- number, need to knock off a second for the filter size
            wind = probas[i:i+filt_size[0], j:j+filt_size[1],:]
            if sig:
                wind = torch.sigmoid(torch.tensor(wind)).numpy()
            if per_species:
                spp = []
                angl = []
                for sp in range(wind.shape[2]):
                    val, ang = filter(wind[:,:,sp], dist_fn, agg_fn)
                    spp.append(val)
                    angl.append(ang)
                convo[i,j] = agg_fn(spp)
                angles[i,j] = agg_fn(angl)
            else:
                blah = filter(wind, dist_fn, agg_fn)
                convo[i,j] = blah
    if angle:
        return convo, angles
    else:
        return convo


def beta_div(predictions, nodata, dtype=np.float64):
# these are the parameters I found produced the best maps. See jupyter notebooks for further exploration
    sig = True
    per_species = False
    agg = 'norm'
    filter = 'combined'
    dist = 'L2'
    filt_size = (3,3)
    angles = False
    return convolve(predictions, filt_size, sig, per_species, dist, filter, agg, angles, nodata)

def gamma_div():
    raise NotImplemented

DIV_METHODS = {
        'alpha': alpha_div,
        'beta': beta_div,
        'gamma': gamma_div,
}
def diversity_raster(rasname, metric, year, base_dir, modelname, nodata=9999, warp=True, **kwargs):

    if metric in DIV_METHODS:
        method = DIV_METHODS[metric]
    elif callable(metric):
        method = metric
    else:
        raise ValueError('Unknown method {0}, must be one of {1} or callable'
                         .format(metric, list(DIV_METHODS.keys())))
    with rasterio.open(rasname) as src:
        # TODO: add mean adjustment :(
        ras = src.read()

        # will not handle if there is nan data in the file
        if "threshold" in kwargs:
            output = method(ras, kwargs)
            dtype = output.dtype
        else:
            output = method(ras, nodata)
            dtype= output.dtype

        if nodata is None:
            raise ValueError(f"you must set a nodata value that matches the save datatype of {output.dtype}! ")
        else:
            if nodata >= output.min() and nodata <= output.max():
                raise ValueError(f"you must set a nodata value that is not in the range of your output array [{output.min()}, {output.max()}]! ")
        kwargs = src.meta.copy()
        kwargs.update({'count' : 1, 'dtype' : output.dtype, 'nodata' : nodata, 'width': output.shape[-1], 'height': output.shape[-2]})
        if warp and src.crs != GBIF_CRS:
            # use rasterio to reproject https://rasterio.readthedocs.io/en/latest/topics/reproject.html
            nnt, wid, hig = calculate_default_transform(src.crs, GBIF_CRS, src.width, src.height, *src.bounds)
            kwargs.update({
                'width': wid,
                'height': hig,
                'transform' : nnt,
                'crs' : GBIF_CRS,
            })
            dest = np.full([hig, wid], nodata, dtype=dtype)
            reproject(output, dest, src_transform=src.transform, src_crs=src.crs, dst_transform=nnt,dst_crs=GBIF_CRS,resampling=Resampling.bilinear)
            output = dest
        fname = f"{base_dir}inference/prediction/{metric}_diversity/{modelname}{rasname.split(modelname)[-1]}"
        if not os.path.exists(fname.rsplit('/',1)[0]): # make directory if needed
            os.makedirs(fname.rsplit('/',1)[0])
        with rasterio.open(fname, 'w',  **kwargs) as dst:
            dst.write(output, 1)
    return fname

def get_alpha_files(shpfile, naip_dir, ca_tifb):
    tif_dir = naip_dir + ca_tifb
    print(tif_dir)
    fnames = []
    for _, fman in shpfile.iterrows():
        fnames.append(Grab_TIFF(fman, tif_dir))
    return fnames

def get_beta_files(shpfile, modelname, base_dir, ca_tifb):
    alph_dir = f"{base_dir}/inference/prediction/alpha_diversity/{modelname}/{ca_tifb}"
    tif_dir = alph_dir + ca_tifb
    print(tif_dir)
    for _, fman in shpfile.iterrows():
        fnames.append(Grab_TIFF(fman, tif_dir))
    print(fnames[0])
    return fnames

  
def predict_joint_array_arbitrary_res(dat, transform, crs, save_file, rasters, b_size, res, spec_names, device, model, modelname, means, std, img_size=utils.IMG_SIZE):
    # TODO: if batch size is too big, this breaks??
    tock = time.time()
    assert len(dat.shape) == 3, 'wrong dimensions for array!'
    swidth, sheight = dat.shape[2], dat.shape[1]
    nwidth, nheight = swidth-img_size, sheight-img_size
    # nwidth, nheight = swidth-(img_size+res), sheight-(img_size+res)
    print(nheight, nwidth)
    i_ind = range(0, nwidth, res)
    j_ind = range(0, nheight, res)
    n_specs = len(spec_names)
    wid, hig = len(i_ind), len(j_ind)
    print(wid, hig)
    # make receiver array
    result = np.full([n_specs, hig, wid],np.nan,  dtype=np.float32)
    ii = 0
    # can't handle a batch size larger than one row
    if b_size > wid:
        raise NotImplementedError

    print(nwidth, nheight, swidth, sheight, wid, hig)
        
        
    with torch.no_grad():
        with tqdm(total=len(i_ind)*(len(j_ind)//b_size)*2, unit="window") as prog:
            # go row by row down raster
            for i in i_ind:
                jj = 0
                # chunk up the row into batches
                for j in utils.chunks(j_ind, b_size):
                    # grab the x / y of top left corner of each image for the climate
                    xys = rasterio.transform.xy(transform, *zip(*[(c,i) for c in j]))
                    # transform x,y into lat/lon crs
                    transf = projTransf.from_crs(crs, GBIF_CRS)
                    xys = list(transf.itransform(zip(*xys)))
                    # and grab the climate from the raster
                    # alas can't use list comprehention
#                     clims = [raster[0, *rasterio.transform.rowcol(transf, x, y)] for raster, transf in rasters for y,x in xys]
                    clims = []
                    for y, x in xys:
                        clim = []
                        for (raster, transf,_) in rasters:
                            row, col = rasterio.transform.rowcol(transf, x, y)
                            clim.append(raster[0,row,col])
                        clims.append(clim)
                    clim = np.vstack(clims)
                    # grab each image at the resolution we prefer from image raster
                    tc = [dat[:, c:c+img_size, i:i+img_size] for c in j]
                    # normalize, scale
                    tc = [utils.scale(t, out_range=(0,1), min_=0, max_=255) for t in tc]
                    tc = [TF.normalize(torch.tensor(t, dtype=torch.float), means, std) for t in tc]
                    # zip together climate and images
                    inputs = [ (t, torch.tensor(c, dtype=torch.float)) for t, c in zip(tc, clim)]
                    inputs = list(zip(*inputs))
                    # for j, i in enumerate(inputs[0]):
                    #     print(j, i.shape)
                    inputs = (torch.stack(inputs[0]), torch.stack(inputs[1]))
                    inputs = ( inputs[0].to(device), inputs[1].to(device))
                    out = model(inputs)
                    out = np.squeeze(out[0].detach().cpu().numpy())
                    # flip outputs to move species axis to 1st axis,
                    # batch axis to row
                    result[:,jj:jj+b_size,ii] = out.T
                    jj +=b_size
                    prog.update(1)
                ii += 1
    prog.close()
    # ugly code to save results to raster    
    out_profile = profiles.DefaultGTiffProfile()
    # get the true bounds of image (accounts for lost pixels on edge)
    west,  north = rasterio.transform.xy(transform, [0],[0])
    east, south = rasterio.transform.xy(transform, [nheight],[nwidth])
    transf = rasterio.transform.from_bounds(west[0], south[0], east[0], north[0], wid, hig)
    bounds = rasterio.transform.array_bounds(hig, wid, transf)
    # get true resolution as well
    out_res = (nwidth/wid, nheight/hig)
    # dst_w = left, dst_n = top
    # trans = Affine.translation(bounds.left, bounds.top) * Affine.scale(out_res[0], -out_res[1])
    out_profile['res'] = out_res
    out_profile['bounds'] = bounds
    out_profile['transform'] = transf
    out_profile['height'] = hig
    out_profile['width'] = wid 
    out_profile['count'] = n_specs
    out_profile['dtype'] = result.dtype
    out_profile.update(BIGTIFF="IF_SAFER")
    print("saving file now")
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    with rasterio.open(save_file, 'w', **out_profile) as dst:
        dst.write(result, range(1,n_specs+1))
        dst.descriptions = spec_names
    tick = time.time()
    print(f"file {save_file.split('/')[-1]} took {(tick-tock)/60} minutes")


# TODO: can't handle resolution lower than the actual resolution used for training
def predict_raster_arbitrary_res(sat_file, save_file, b_size, res, spec_names, device, model, modelname, means, std, img_size=utils.IMG_SIZE):
    # TODO: if batch size is too big, this breaks??
    tock = time.time()
    sat  = rasterio.open(sat_file)
    dat = sat.read()
    swidth, sheight = sat.width, sat.height
    # leave off the last pixels for which we don't have a full 256 image for
    nwidth, nheight = swidth-img_size, sheight-img_size
    i_ind = range(0, nwidth, res)
    j_ind = range(0, nheight, res)
    n_specs = len(spec_names)
    wid, hig = len(i_ind), len(j_ind)
    # make receiver array
    result = np.full([n_specs, hig, wid],np.nan,  dtype=np.float32)
    ii = 0
    # can't handle a batch size larger than one row
    if b_size > wid:
        raise NotImplementedError

    with torch.no_grad():
        with tqdm(total=len(i_ind)*(len(j_ind)//b_size)*2, unit="window") as prog:
            # go row by row down raster
            for i in i_ind:
                jj = 0
                # chunk up the row into batches
                for j in utils.chunks(j_ind, b_size):
                    # grab each image at the resolution we prefer from image raster
                    tc = [dat[:, c:c+img_size, i:i+img_size] for c in j]
                    # normalize, scale
                    tc = [utils.scale(t, out_range=(0,1), min_=0, max_=255) for t in tc]
                    tc = [TF.normalize(torch.tensor(t), means, std) for t in tc]
                    tc = torch.stack(tc)
                    tc = tc.to(device)
                    out = model(tc.float())
                    if len(out) == 3:
                        # handles models that predict all taxa versus
                        # just species
                        out = np.squeeze(out[0].detach().cpu().numpy())
                    else:
                        out = np.squeeze(out.detach().cpu().numpy())
                    # flip outputs to move species axis to 1st axis,
                    # batch axis to row
                    result[:,jj:jj+b_size,ii] = out.T
                    jj +=b_size
                    prog.update(1)
                ii += 1
    prog.close()
    # ugly code to save results to raster    
    out_profile = sat.profile
    # get the true bounds of image (accounts for lost pixels on edge)
    # TODO: check it is west, north not other way around!
    west,  north = rasterio.transform.xy(sat.transform, [0],[0])
    east, south = rasterio.transform.xy(sat.transform, [nheight],[nwidth])
    transf = rasterio.transform.from_bounds(west[0], south[0], east[0], north[0], wid, hig)
    bounds = rasterio.transform.array_bounds(hig, wid, transf)
    # get true resolution as well
    out_res = (nwidth/wid, nheight/hig)
    # dst_w = left, dst_n = top
    # trans = Affine.translation(bounds.left, bounds.top) * Affine.scale(out_res[0], -out_res[1])
    out_profile['res'] = out_res
    out_profile['bounds'] = bounds
    out_profile['transform'] = transf
    out_profile['height'] = hig
    out_profile['width'] = wid 
    out_profile['count'] = n_specs
    out_profile['dtype'] = result.dtype
    out_profile.update(BIGTIFF="IF_SAFER")
    print("saving file now")
    sat.close()
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    with rasterio.open(save_file, 'w', **out_profile) as dst:
        dst.write(result, range(1,n_specs+1))
        dst.descriptions = spec_names
    tick = time.time()
    print(f"file {save_file.split('/')[-1]} took {(tick-tock)/60} minutes")

    
def predict_joint_raster_arbitrary_res(sat_file, save_file, rasters, b_size, res, spec_names, device, model, modelname, means, std, img_size=utils.IMG_SIZE):
    # TODO: if batch size is too big, this breaks??
    tock = time.time()
    sat  = rasterio.open(sat_file)
    dat = sat.read()
    swidth, sheight = sat.width, sat.height
    # leave off the last pixels for which we don't have a full 256 image for
    # and make sure rounds to resolution!
    # nwidth = swidth-(swidth%img_size)
    # nwidth = nwidth-(nwidth%res)
    # nheight = sheight-(sheight%img_size)
    # nheight = nheight-(nheight%res)
    nwidth, nheight = swidth-img_size, sheight-img_size
    i_ind = range(0, nwidth, res)
    j_ind = range(0, nheight, res)
    n_specs = len(spec_names)
    wid, hig = len(i_ind), len(j_ind)
    # make receiver array
    result = np.full([n_specs, hig, wid],np.nan,  dtype=np.float32)
    ii = 0
    # can't handle a batch size larger than one row
    if b_size > wid:
        raise NotImplementedError

    with torch.no_grad():
        with tqdm(total=len(i_ind)*(len(j_ind)//b_size)*2, unit="window") as prog:
            # go row by row down raster
            for i in i_ind:
                jj = 0
                # chunk up the row into batches
                for j in utils.chunks(j_ind, b_size):
                    # grab the x / y of top left corner of each image for the climate
                    xys = rasterio.transform.xy(sat.transform, *zip(*[(c,i) for c in j]))
                    # transform x,y into lat/lon crs
                    transf = projTransf.from_crs(sat.crs, GBIF_CRS)
                    xys = list(transf.itransform(zip(*xys)))
                    # and grab the climate from the raster
                    # alas can't use list comprehention
#                     clims = [raster[0, *rasterio.transform.rowcol(transf, x, y)] for raster, transf in rasters for y,x in xys]
                    clims = []
                    for y, x in xys:
                        clim = []
                        for (raster, transf,_, i_) in rasters:
                            row, col = rasterio.transform.rowcol(transf, x, y)
                            clim.append(raster[0,row,col])
                        clims.append(clim)
                    clim = np.vstack(clims)
                    # grab each image at the resolution we prefer from image raster
                    tc = [dat[:, c:c+img_size, i:i+img_size] for c in j]
                    # normalize, scale
                    tc = [utils.scale(t, out_range=(0,1), min_=0, max_=255) for t in tc]
                    tc = [TF.normalize(torch.tensor(t, dtype=torch.float), means, std) for t in tc]
                    # zip together climate and images
                    inputs = [ (t, torch.tensor(c, dtype=torch.float)) for t, c in zip(tc, clim)]
                    inputs = list(zip(*inputs))
                    inputs = (torch.stack(inputs[0]), torch.stack(inputs[1]))
                    inputs = ( inputs[0].to(device), inputs[1].to(device))
                    out = model(inputs)
                    out = np.squeeze(out[0].detach().cpu().numpy())
                    # flip outputs to move species axis to 1st axis,
                    # batch axis to row
                    result[:,jj:jj+b_size,ii] = out.T
                    jj +=b_size
                    prog.update(1)
                ii += 1
    prog.close()
    # ugly code to save results to raster    
    out_profile = sat.profile
    # get the true bounds of image (accounts for lost pixels on edge)
    west,  north = rasterio.transform.xy(sat.transform, [0],[0])
    east, south = rasterio.transform.xy(sat.transform, [nheight],[nwidth])
    transf = rasterio.transform.from_bounds(west[0], south[0], east[0], north[0], wid, hig)
    bounds = rasterio.transform.array_bounds(hig, wid, transf)
    # get true resolution as well
    out_res = (nwidth/wid, nheight/hig)
    # dst_w = left, dst_n = top
    # trans = Affine.translation(bounds.left, bounds.top) * Affine.scale(out_res[0], -out_res[1])
    out_profile['res'] = out_res
    out_profile['bounds'] = bounds
    out_profile['transform'] = transf
    out_profile['height'] = hig
    out_profile['width'] = wid 
    out_profile['count'] = n_specs
    out_profile['dtype'] = result.dtype
    out_profile.update(BIGTIFF="IF_SAFER")
    print("saving file now")
    sat.close()
    if not os.path.exists(os.path.dirname(save_file)):
        os.makedirs(os.path.dirname(save_file))
    with rasterio.open(save_file, 'w', **out_profile) as dst:
        dst.write(result, range(1,n_specs+1))
        dst.descriptions = spec_names
    tick = time.time()
    print(f"file {save_file.split('/')[-1]} took {(tick-tock)/60} minutes")