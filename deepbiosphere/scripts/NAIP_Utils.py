import torch
import os
import logging
import math
from pathlib import Path
import warnings
from rasterio import Affine, MemoryFile
from rasterio.profiles import DefaultGTiffProfile
from rasterio.warp import calculate_default_transform, reproject, Resampling
import numpy as np

import rasterio
from rasterio.crs import CRS
import rasterio.warp
import rasterio.shutil
from rasterio.enums import Resampling
from rasterio import shutil as rio_shutil
from rasterio.vrt import WarpedVRT
from rasterio.coords import disjoint_bounds
from rasterio.enums import Resampling
from rasterio.windows import Window
from deepbiosphere.scripts import new_window
from deepbiosphere.scripts import GEOCLEF_Utils as utils
from rasterio.transform import Affine
import rasterio.transform as transforms
import matplotlib as mpl
import matplotlib.patches as patches
import math
import matplotlib.cm as cm
import matplotlib as mpl
import time
# deepbio packages
from deepbiosphere.scripts.GEOCLEF_Config import paths
import deepbiosphere.scripts.GEOCLEF_Config as config
from deepbiosphere.scripts.GEOCLEF_Run import  setup_dataset
import deepbiosphere.scripts.GEOCLEF_Utils as utils
import deepbiosphere.scripts.GEOCLEF_CNN as cnn
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, MultiPolygon, LineString
import shapely.speedups

# Standard packages
import tempfile
import warnings
import urllib
import shutil
import os

# Less standard, but still pip- or conda-installable
import matplotlib.pyplot as plt
import numpy as np
import rasterio
import re # regex
import rtree
import shapely
import pickle

# pip install progressbar2, not progressbar
import progressbar

from geopy.geocoders import Nominatim
from rasterio.merge import merge
from tqdm import tqdm


import fiona
import fiona.transform
import requests
import json
import torch
import numpy as np


## fields
# not all the NAIP are teh same coorediate reference system
# this is WGS84, what the VRT are converted to
NAIP_CRS='EPSG:4326'

class DownloadProgressBar():
    """
    https://stackoverflow.com/questions/37748105/how-to-use-progressbar-module-with-urlretrieve
    """

    def __init__(self):
        self.pbar = None

    def __call__(self, block_num, block_size, total_size):
        if not self.pbar:
            self.pbar = progressbar.ProgressBar(max_value=total_size)
            self.pbar.start()

        downloaded = block_num * block_size
        if downloaded < total_size:
            self.pbar.update(downloaded)
        else:
            self.pbar.finish()

# very important class!
class NAIPTileIndex:
    """
    Utility class for performing NAIP tile lookups by location.
    """

    tile_rtree = None
    tile_index = None
    base_path = None

    def __init__(self, index_blob_root, index_files, base_path=None, temp_dir=None):

        temp_dir = os.path.join(tempfile.gettempdir(),'naip') if temp_dir is None else temp_dir
        if base_path is None:

            base_path = temp_dir
            os.makedirs(base_path,exist_ok=True)

        for file_path in index_files:
            download_url(index_blob_root + file_path, base_path + '/' + file_path,
                         progress_updater=DownloadProgressBar())

        self.base_path = base_path
        # tile_rtree is an rtree that stores I believe bounding boxes for the tifs
        self.tile_rtree = rtree.index.Index(base_path + "/tile_index")
        self.tile_index = pickle.load(open(base_path  + "/tiles.p", "rb"))


    def lookup_tile(self, lat, lon):
        """"
        Given a lat/lon coordinate pair, return the list of NAIP tiles that contain
        that location.

        Returns a list of COG file paths.
        """

        point = shapely.geometry.Point(float(lon),float(lat))
        intersected_indices = list(self.tile_rtree.intersection(point.bounds)) # oh wow so this rtree does ALL the heavy lifting, phew....
        intersected_files = []
        tile_intersection = False

        for idx in intersected_indices:

            intersected_file = self.tile_index[idx][0]
            intersected_geom = self.tile_index[idx][1]
            if intersected_geom.contains(point): # Ohh I see, so it might be an rtree miss so still have to check to be sure
                tile_intersection = True
                intersected_files.append(intersected_file)

        if not tile_intersection and len(intersected_indices) > 0: # How can this be??
            print('''Error: there are overlaps with tile index,
                      but no tile completely contains selection''')
            return None
        elif len(intersected_files) <= 0:
            print("No tile intersections")
            return None
        else:
            return intersected_files


def download_url(url, destination_filename=None, progress_updater=None, force_download=False):
    """
    Download a URL to a temporary file
    """

    # This is not intended to guarantee uniqueness, we just know it happens to guarantee
    # uniqueness for this application.
    if destination_filename is None:
        url_as_filename = url.replace('://', '_').replace('/', '_')
        destination_filename = \
            os.path.join(temp_dir,url_as_filename)
    if (not force_download) and (os.path.isfile(destination_filename)):
        print('Bypassing download of already-downloaded file {}'.format(os.path.basename(url)))
        return destination_filename
    print('Downloading file {} to {}'.format(os.path.basename(url),destination_filename),end='')
    urllib.request.urlretrieve(url, destination_filename, progress_updater)
    assert(os.path.isfile(destination_filename))
    nBytes = os.path.getsize(destination_filename)
    print('...done, {} bytes.'.format(nBytes))
    return destination_filename


def display_naip_tile(filename):
    """
    Display a NAIP tile using rasterio.
    """

    # NAIP tiles are enormous; downsize for plotting in this notebook
    dsfactor = 10

    with rasterio.open(filename) as raster:

        # NAIP imagery has four channels: R, G, B, IR
        #
        # Stack RGB channels into an image; we won't try to render the IR channel
        #
        # rasterio uses 1-based indexing for channels.
        h = int(raster.height/dsfactor)
        w = int(raster.width/dsfactor)
        print('Resampling to {},{}'.format(h,w))
        r = raster.read(1, out_shape=(1, h, w))
        g = raster.read(2, out_shape=(1, h, w))
        b = raster.read(3, out_shape=(1, h, w))

    rgb = np.dstack((r,g,b))
    fig = plt.figure(figsize=(7.5, 7.5), dpi=100, edgecolor='k')
    plt.imshow(rgb)
    raster.close()

def setup_NAIPIndex(blob_root=paths.BLOB_ROOT, crs=NAIP_CRS, index_loc=None):

        index_files = ['tile_index.dat', 'tile_index.idx', 'tiles.p'] # these are the files that tell you which tiles are where
        index_blob_root = re.sub('/naip$','/naip-index/rtree/',blob_root)
        temp_dir = os.path.join(tempfile.gettempdir(),'naip') if index_loc is None else index_loc
        os.makedirs(temp_dir,exist_ok=True)
        # Spatial index that maps lat/lon to NAIP tiles; we'll load this when we first
        # need to access it.
        index = None
        warnings.filterwarnings("ignore")
        return NAIPTileIndex(index_blob_root, index_files,  temp_dir)

# types for type hints
NAIP_shpfile  = gpd.geodataframe.GeoDataFrame
Point = shapely.geometry.Point

def Load_Cali_Bounds(base_dir : str):
     us1 = gpd.read_file(base_dir + 'us_shapefiles/gadm36_USA_1.shp') # the state's shapefiles
     ca = us1[us1.NAME_1 == 'California']
     return ca

def Load_NAIP_Bounds(base_dir : str, state: str, year : str):
    return gpd.read_file(
            f"{base_dir}{state}/{year}/{state}_shpfl_{year}/naip_3_{year[2:4]}_1_1_{state}.shp")

def Find_Rasters_Polygon(gdf : NAIP_shpfile, poly : Polygon, bands : list, base_dir : str,  res : float = None, ftype : str = "vrt"):
    check_bands(bands)
    null = Polygon()
    tt = gdf.intersection(poly)
    rasters =  gdf[tt != null]
    rasters = [f"{base_dir}/{fman.APFONAME[:5]}/{'_'.join(fman.FileName.split('_')[:-1])}.{ftype}" for _, fman in rasters.iterrows()]
    if len(rasters) > 1:
        return merge(rasters, res=res, indexes=bands)
    elif len(rasters) == 0:
        print("no matching rasters found")
    else:
        # TODO: handle resolution
        with rasterio.open(rasters[0]) as r:
            ras, tran = r.read(indexes=bands), r.transform
            return ras, tran

def check_bands(bands):
    if bands is None:
        raise ValueError("bands should be specificed! If you want all bands, return a list with all bands, else GDAL reverts to an incorrect driver that interprets the 4th band as alpha")

def Find_Rasters_Point(gdf : NAIP_shpfile, point : Point, bands : list, base_dir  : str, res : float = None, ftype : str = 'vrt'):
    check_bands(bands)
    rasters = gdf[gdf.contains(point)]
    rasters = [f"{base_dir}/{fman.APFONAME[:5]}/{'_'.join(fman.FileName.split('_')[:-1])}.{ftype}" for _, fman in rasters.iterrows()]
    if len(rasters) > 1:
        return
def merge(
    datasets,
    bounds=None,
    res=None,
    nodata=None,
    dtype=None,
    precision=None,
    indexes=None,
    output_count=None,
    resampling=Resampling.nearest,
    method="first",
    target_aligned_pixels=False,
    dst_path=None,
    dst_kwds=None,
):
    if method in MERGE_METHODS:
        copyto = MERGE_METHODS[method]
    elif callable(method):
        copyto = method
    else:
        raise ValueError('Unknown method {0}, must be one of {1} or callable'
                         .format(method, list(MERGE_METHODS.keys())))

    # Create a dataset_opener object to use in several places in this function.
    if isinstance(datasets[0], str) or isinstance(datasets[0], Path):
        dataset_opener = rasterio.open
    else:

        @contextmanager
        def nullcontext(obj):
            try:
                yield obj
            finally:
                pass

        dataset_opener = nullcontext

    check_bands(indexes)
    with dataset_opener(datasets[0]) as first:
        first_profile = first.profile
        first_res = first.res
        nodataval = first.nodatavals[0]
        dt = first.dtypes[0]

        if indexes is None:
            src_count = first.count
        elif isinstance(indexes, int):
            src_count = indexes
        else:
            src_count = len(indexes)

        try:
            first_colormap = first.colormap(1)
        except ValueError:
            first_colormap = None

    if not output_count:
        output_count = src_count

    # Extent from option or extent of all inputs
    if bounds:
        dst_w, dst_s, dst_e, dst_n = bounds
    else:
        # scan input files
        xs = []
        ys = []
        for dataset in datasets:
            with dataset_opener(dataset) as src:
                left, bottom, right, top = src.bounds
            xs.extend([left, right])
            ys.extend([bottom, top])
        dst_w, dst_s, dst_e, dst_n = min(xs), min(ys), max(xs), max(ys)

    # Resolution/pixel size
    if not res:
        res = first_res
    elif not np.iterable(res):
        res = (res, res)
    elif len(res) == 1:
        res = (res[0], res[0])

    if target_aligned_pixels:
        dst_w = math.floor(dst_w / res[0]) * res[0]
        dst_e = math.ceil(dst_e / res[0]) * res[0]
        dst_s = math.floor(dst_s / res[1]) * res[1]
        dst_n = math.ceil(dst_n / res[1]) * res[1]

    # Compute output array shape. We guarantee it will cover the output
    # bounds completely
# because round() has weird behavior, instead we're going to
# round up the size of the array always and slightly
# stretch the rasters to fill it
    output_width = math.ceil((dst_e - dst_w) / res[0])
    output_height = math.ceil((dst_n - dst_s) / res[1])
    output_transform = Affine.translation(dst_w, dst_n) * Affine.scale(res[0], -res[1])

    if dtype is not None:
        dt = dtype

    out_profile = first_profile
    out_profile.update(**(dst_kwds or {}))

    out_profile["transform"] = output_transform
    out_profile["height"] = output_height
    out_profile["width"] = output_width
    out_profile["count"] = output_count
    if nodata is not None:
        out_profile["nodata"] = nodata

    # create destination array
    dest = np.zeros((output_count, output_height, output_width), dtype=dt)
    if nodata is not None:
        nodataval = nodata

    if nodataval is not None:
        # Only fill if the nodataval is within dtype's range
        inrange = False
        if np.issubdtype(dt, np.integer):
            info = np.iinfo(dt)
            inrange = (info.min <= nodataval <= info.max)
        elif np.issubdtype(dt, np.floating):
            if math.isnan(nodataval):
                inrange = True
            else:
                info = np.finfo(dt)
                inrange = (info.min <= nodataval <= info.max)
        if inrange:
            dest.fill(nodataval)
        else:
            warnings.warn(
                "The nodata value, %s, is beyond the valid "
                "range of the chosen data type, %s. Consider overriding it "
                "using the --nodata option for better results." % (
                    nodataval, dt))
    else:
        nodataval = 0
    for idx, dataset in enumerate(datasets):
        with dataset_opener(dataset) as src:
            # Real World (tm) use of boundless reads.
            # This approach uses the maximum amount of memory to solve the

            if disjoint_bounds((dst_w, dst_s, dst_e, dst_n), src.bounds):
                continue
# 1. Compute spatial intersection of destination and source
            src_w, src_s, src_e, src_n = src.bounds
            # so src.bounds is the extent of the raster
            # in the coordinate system of the raster (ie: lat/lon degree extent)
            # so, what this below chunk says is that if the current raster is bigger than the bounds outlined
# is outside of your bounds, only take to the b ounds
            int_w = src_w if src_w > dst_w else dst_w
            int_s = src_s if src_s > dst_s else dst_s
            int_e = src_e if src_e < dst_e else dst_e
            int_n = src_n if src_n < dst_n else dst_n
            # so then this next section just takes these extents (ignoring the boudns ig?)
            # plus the affine transform and tries to line up the pixels
            # 2. Compute the source window
            # because we wanna be safe and avoid padding errors
# we take the floor of the min coordinate and the ceil of the max
# coordinate so that the downsampling of the raster meets the
# largest bounding box it can fit. This may lead to some overlap
# on the edges, and currently we're using painter's overlap to
# deal with this inconsistency. It's such a boundary thing that it
# shouldn't be an issue though...
            src_window = from_bounds(
                int_w, int_s, int_e, int_n, src.transform, precision=precision
            )

            # 3. Compute the destination window
            dst_window = from_bounds(
                int_w, int_s, int_e, int_n, output_transform, precision=precision
            )
            # 4. Read data in source window into temp
# no longer convinced all this junk is relevant below
# kinda problematic anyway I don't think it's relevant
# basically this code rounds ah and there may be offending
# problems with the rounding as well, rip
# yeah let's avoid like the plague...
            temp_shape = (src_count, dst_window.height, dst_window.width)
            temp_src = src.read(
                out_shape=temp_shape,
                window=src_window,
                boundless=False,
                masked=True,
                indexes=indexes,
                resampling=resampling,
            )

            # 5. Copy elements of temp into dest
            region = dest[:, dst_window.row_off : dst_window.row_off + dst_window.height, dst_window.col_off : dst_window.col_off+ dst_window.width]

            # okay so this region mask will select for regions that have nooo data fill so far
            if math.isnan(nodataval):
                region_mask = np.isnan(region)
            elif np.issubdtype(region.dtype, np.floating):
                region_mask = np.isclose(region, nodataval)
            else:
                region_mask = region == nodataval

        temp_mask = np.ma.getmask(temp_src)
      #  so, copyto is a special case of np.copyto I think
      #  that either takes the current or previous or min or max
      #  value of that set. I don't know what the default is
        copyto(region, temp_src, region_mask, temp_mask, index=idx, roff=dst_window.row_off, coff=dst_window.col_off)

    if dst_path is None:
        return dest, output_transform

    else:
        with rasterio.open(dst_path, "w", **out_profile) as dst:
            dst.write(dest)
            if first_colormap:
                dst.write_colormap(1, first_colormap)
def from_bounds(
    left, bottom, right, top, transform=None, height=None, width=None, precision=None
):
    """Get the window corresponding to the bounding coordinates.

    Parameters
    ----------
    left: float, required
        Left (west) bounding coordinates
    bottom: float, required
        Bottom (south) bounding coordinates
    right: float, required
        Right (east) bounding coordinates
    top: float, required
        Top (north) bounding coordinates
    transform: Affine, required
        Affine transform matrix.
    height: int, required
        Number of rows of the window.
    width: int, required
        Number of columns of the window.
    precision: int or float, optional
        An integer number of decimal points of precision when computing
        inverse transform, or an absolute float precision.

    Returns
    -------
    Window
        A new Window.

    Raises
    ------
    WindowError
        If a window can't be calculated.

    """
    if not isinstance(transform, Affine):
        raise WindowError("A transform object is required to calculate the window")

    if (right - left) / transform.a < 0:
        raise WindowError("Bounds and transform are inconsistent")

    if (bottom - top) / transform.e < 0:
        raise WindowError("Bounds and transform are inconsistent")

    rows, cols = transforms.rowcol(
        transform,
        [left, right, right, left],
        [top, top, bottom, bottom],
        op=float,
        precision=precision,
    )
# make the window larger than needed by taking floor and ceil
    row_start, row_stop = max(0, math.floor(min(rows))), max(0, math.ceil(max(rows)))
    col_start, col_stop = math.floor(min(cols)), math.ceil(max(cols))

    return Window(
        col_off=col_start,
        row_off=row_start,
        width=max(col_stop - col_start, 0.0),
        height=max(row_stop - row_start, 0.0),
    )

def Mask_Raster(raster : np.array, trans, polygon, crs=NAIP_CRS):
        kwargs = {
            'transform' :trans,
            'driver': 'GTiff',
            'height' : raster.shape[1],
            'width' : raster.shape[2]
        }
        with MemoryFile() as memfile:
            with memfile.open(**DefaultGTiffProfile(count=raster.shape[0], width = raster.shape[2], height = raster.shape[1],  crs=crs, transform=trans)) as dataset: # Open as DatasetWriter
                profile = dataset.profile
                profile.update(**kwargs)
                dataset.write(rasters)
                del rasters
            with memfile.open(transform=trans) as dataset:  # Reopen as DatasetReader
                cropped, trans = mask(dataset, polygon.geometry, invert=False, crop=True)
                return cropped, trans

# tif_dir should be the base_dir plus the year and acquisiiton directory
def Warp_VRT(vrt_dir : str, tif_dir : str,  shpfile : NAIP_shpfile, dst_crs = NAIP_CRS):
    for i, fman in shpfile.iterrows():
        # build local filepath
        fname = Grab_TIFF(fman, tif_dir)
        # f"{tif_dir}/{fman.APFONAME[:5]}/{'_'.join(fman.FileName.split('_')[:-1])}.tif"
# I dont think I actually need this
        dst_bounds = fman.geometry.bounds
        with rasterio.open(fname) as src:
            dst_height = src.profile['height']
            dst_width =  src.profile['width']
            # WGS84 transform
            left, bottom, right, top = dst_bounds
            xres = (right - left) / dst_width
            yres = (top - bottom) / dst_height
            dst_transform = affine.Affine(xres, 0.0, left,
                                          0.0, -yres, top)
            vrt_options = {
                'resampling': Resampling.bilinear,
                'crs': dst_crs,
                'transform': dst_transform,
                'height': dst_height,
                'width': dst_width,
                'add_alpha' : False,
                'nodata' : 0,
                'warp_mem_limit' : 0.0
            }
            with WarpedVRT(src, **vrt_options) as vrt:
                # recreate the same file directory structure as NAIP
                ndir = f"{vrt_dir}/{fman.APFONAME[:5]}"
                None if os.path.isdir(ndir) else os.makedirs(ndir) # sneaky one liner make directory
                fpath = ndir + f"/{'_'.join(fman.FileName.split('_')[:-1])}.vrt" # now make full filename
                rio_shutil.copy(vrt, fpath, driver='VRT')
    tick = time.time()
    print(f"took {(tick-tock)/60} minutes to convert to VRT")



def predict_raster_list(device_no, tiffs, modelname, res, year, model_pth, cfg_pth, base_dir=paths.AZURE_DIR):
    # TODO: remove this garbage belwo
    #if modelname == 'old_tresnet_satonly':
    if "old" in modelname:
            # basedire for config and dataset are different
            basedir = "/home/leg/deepbiosphere/GeoCELF2020/"
    else:
        basedir = base_dir
    if device_no == 'cpu':
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device_no}")
        torch.cuda.set_device(device_no)
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
    model = cnn.TResNet_M(params.params.pretrained, num_spec, num_gen, num_fam, basedir)
    model.load_state_dict(state['model_state_dict'], strict=True)
    model = model.to(device)
    model.eval();
    spec_names = tuple(daset.inv_spec.values())
    # figure out batch size
    batchsize = batchsized(device, tiffs[0], model,params.params.batch_size, res, num_spec) # params.params.batch_size
    for raster in tqdm(tiffs):
        file = predict_raster(raster, model, batchsize, res, year, base_dir, modelname, num_spec, device, spec_names)


def predict_raster(rasname, model, batchsize, res, year, base_dir, modelname, num_spec, device, spec_names):
    with rasterio.open(rasname) as src:
        ras = src.read()
        width, height = src.width, src.height
        num_w, num_h  = width//res, height//res
        clean_w, clean_h = num_w *res, num_h*res
        new_w = num_w + 1 if width % res != 0 else num_w
        new_h = num_h + 1 if height % res != 0 else num_h
        output = np.zeros([num_spec, new_h, new_w])
        images = []
        # grab all the locations that normally fit
        for i in range(0, clean_w, res):
            for j in range(0, clean_h, res):
                ii, jj = int(i/res), int(j/res)
                window = ras[:, j:j+res, i:i+res]
                # so, the subtraction doesn't work becuase of the uint8 datatype, so taking out for now
                images.append((window, [ii,jj]))
        # add a row if there are pixels that bleed over the bottom
        if height % res != 0:
            for i in range(0, clean_w, res):
                ii, jj = int(i/res), height//res
                window = ras[:, height-res:, i:i+res]
                images.append((window, [ii,jj]))
        # add a row if there are pixels that bleed over the edge
        if width % res != 0:
            for j in range(0, clean_h, res):
                ii, jj = width//res, int(j/res)
                window = ras[:, j:j+res , width-res:]
                images.append((window, [ii,jj]))
        # finish adding bleed prediction
        if width % res != 0 and height % res != 0:
            ii, jj = width//res, height//res
            window = ras[:, height-res: , width-res:]
            images.append((window, [ii,jj]))
        # now chunk the images array by batchsize
        for chunk in utils.chunks(images, batchsize):
            locs = [l for _, l in chunk]
            batch = torch.stack([torch.tensor(c, device=device) for c, _ in chunk], dim=0)
            out = model(batch.float())[0].cpu().detach().numpy() # TODO: these predictions are bunk for the old model bc of the issue with setting things...
            for (loc, outt) in zip(locs, out):
                output[:, loc[1], loc[0]] = outt

        # there's a chance that there's some tomfoolery in the order of
        # bounds and w/h, but I'm going to trust that it works for now
        nt = rasterio.transform.from_bounds(*src.bounds, new_w, new_h)
        # look at src.bounds, nt, below as well
        nnt, wid, hig = calculate_default_transform(src.crs, NAIP_CRS, new_w, new_h, *src.bounds)
        kwargs = src.meta.copy()
        kwargs.update({
            'width': wid,
            'height': hig,
            'count' : num_spec,
            'transform' : nnt,
            'crs' : NAIP_CRS,
            'dtype' : 'float32'
            # so I ran torchy sig with both 64 and 32 and nothing seemed to change so ignoring for now
        })
        fname = f"{base_dir}inference/prediction/{modelname}/{rasname.split(paths.NAIP_BASE)[-1]}" # and steal filename from NAIP
        if not os.path.exists(fname.rsplit('/',1)[0]): # make directory if needed
            os.makedirs(fname.rsplit('/',1)[0])
        with rasterio.open(fname, 'w', **kwargs) as dst:
            # use rasterio to reproject https://rasterio.readthedocs.io/en/latest/topics/reproject.html
            output = np.float32(output)
            dest = np.zeros([num_spec, hig, wid])
            reproject(output, dest, src_transform=nt, src_crs=src.crs, dst_transform=nnt,dst_crs=NAIP_CRS,resampling=Resampling.bilinear)
            dst.write(dest, range(1,len(spec_names)+1))
            dst.descriptions = spec_names
    return fname



# might use, idk cuda GPU mem is weird
def batchsized(device, rasname, model,base_size, res, num_spec):
    with rasterio.open(rasname) as src:
        ras = src.read()
        width, height = src.width, src.height
        num_w, num_h  = width//res, height//res
        clean_w, clean_h = num_w *res, num_h*res
        new_w = num_w + 1 if width % res != 0 else num_w
        new_h = num_h + 1 if height % res != 0 else num_h
        output = np.zeros([num_spec, new_h, new_w])
        images = []
        for i in range(0, clean_w, res):
            for j in range(0, clean_h, res):
                ii, jj = int(i/res), int(j/res)
                window = ras[:, j:j+res, i:i+res]
                images.append((window, [ii,jj]))
        if height % res != 0:
            for i in range(0, clean_w, res):
                ii, jj = int(i/res), height//res
                window = ras[:, height-res:, i:i+res]
                images.append((window, [ii,jj]))
        if width % res != 0:
            for j in range(0, clean_h, res):
                ii, jj = width//res, int(j/res)
                window = ras[:, j:j+res , width-res:]
                images.append((window, [ii,jj]))
        if width % res != 0 and height % res != 0:
            ii, jj = width//res, height//res
            window = ras[:, height-res: , width-res:]
            images.append((window, [ii,jj]))

    for i in range(base_size, 1, -5):
        good = True
        for j, chunk in enumerate(utils.chunks(images, i)):
            batch = torch.stack([torch.tensor(c, device=device) for c, _ in chunk], dim=0)
            try:
                model(batch.float())[0].cpu().detach().numpy()
                # print(f"best batch size is {i}")
            except:
                good = False
                print(f"{i} didn't work")
                break
            if j >= 5: # don't need to loop through everything, likely can hold it if  this  true
                break
        if good:
            print(f"batch size is {i}")
            return i


