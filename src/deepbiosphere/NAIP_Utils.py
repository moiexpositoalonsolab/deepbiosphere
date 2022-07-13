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
from scipy.spatial import distance
from functools import reduce
import rasterio
from rasterio.mask import mask
from rasterio.crs import CRS
import rasterio.warp
import rasterio.shutil
from rasterio.enums import Resampling
from rasterio import shutil as rio_shutil
from rasterio.vrt import WarpedVRT
from rasterio.coords import disjoint_bounds
from rasterio.enums import Resampling
from rasterio.windows import Window
from deepbiosphere import new_window
import deepbiosphere.GEOCLEF_Utils as utils
from rasterio.transform import Affine
import rasterio.transform as transforms
import matplotlib as mpl
import matplotlib.patches as patches
import math
import matplotlib.cm as cm
import matplotlib as mpl
import time
# deepbio packages
from deepbiosphere.Utils import paths
import deepbiosphere.Utils as utils
import deepbiosphere.Models as mods
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
# use below as the default
M_CRS_1 = 'EPSG:26911'
M_CRS_2 = 'EPSG:26910'
IMG_SIZE = 256
ALPHA_NODATA = 9999


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
#             print(index_blob_root + file_path, base_path + '/' + file_path)
            download_url(index_blob_root + file_path, base_path + '/' + file_path)

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
    # make directory if not yet made
    dir = destination_filename.rsplit('/', 1)[0]
    print("dir is ", dir)
    if not os.path.exists(dir):
        print(f"making {destination_filename.rsplit('/', 1)[0]}")
        os.makedirs(dir) 
    print('Downloading file {} to {}'.format(os.path.basename(url),destination_filename),end='')
    urllib.request.urlretrieve(url, destination_filename, progress_updater)
    nBytes = os.path.getsize(destination_filename)
    print('...done, {} bytes.'.format(nBytes))
    return destination_filename


def download_urls_batch(urls):
    i = 0
    for url, dest in urls:
        print("on url, ", i)
        i += 1
        _ = download_url(url, dest, None, False)

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
        index_blob_root = f"{blob_root}naip-index/rtree/"
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
    if year == '2012':
        return gpd(
            f"{base_dir}{state}/{year}/{state}_shpfl_{year}/naip_3_{year[2:4]}_1_1_{state}.shp")
    elif year == '2014':
        return gpd.read_file(
            f"{base_dir}{state}/{year}/{state}_shpfl_{year}/naip_3_{year[2:4]}_3_1_{state}.shp")


def get_Bandnames(gdf : NAIP_shpfile, bands: list, base_dir : str, ftype : str = 'tif'):
    check_bands(bands)
    f = f"{base_dir}/{gdf.iloc[0].APFONAME[:5]}/{'_'.join(gdf.iloc[0].FileName.split('_')[:-1])}.{ftype}"
    with rasterio.open(f) as r:
        return [r.descriptions[i-1] for i in bands]


def Find_Rasters_Polygon(gdf : NAIP_shpfile, poly : Polygon, bands : list, base_dir : str,  res : float = None, ftype : str = "vrt", nodata=None):
    check_bands(bands)
    null = Polygon()
    tt = gdf.intersection(poly)
    rasters =  gdf[tt != null]
    print(f"{len(rasters)} total rasters")
    rasters = [f"{base_dir}/{fman.APFONAME[:5]}/{'_'.join(fman.FileName.split('_')[:-1])}.{ftype}" for _, fman in rasters.iterrows()]
    if len(rasters) > 1:
        return merge(rasters, res=res, indexes=bands, nodata=nodata)
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

def copy_first(merged_data, new_data, merged_mask, new_mask, **kwargs):
    mask = np.empty_like(merged_mask, dtype="bool")
    np.logical_not(new_mask, out=mask)
    np.logical_and(merged_mask, mask, out=mask)
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")


def copy_last(merged_data, new_data, merged_mask, new_mask, **kwargs):
    mask = np.empty_like(merged_mask, dtype="bool")
    np.logical_not(new_mask, out=mask)
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")


def copy_min(merged_data, new_data, merged_mask, new_mask, **kwargs):
    mask = np.empty_like(merged_mask, dtype="bool")
    np.logical_or(merged_mask, new_mask, out=mask)
    np.logical_not(mask, out=mask)
    np.minimum(merged_data, new_data, out=merged_data, where=mask)
    np.logical_not(new_mask, out=mask)
    np.logical_and(merged_mask, mask, out=mask)
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")


def copy_max(merged_data, new_data, merged_mask, new_mask, **kwargs):
    mask = np.empty_like(merged_mask, dtype="bool")
    np.logical_or(merged_mask, new_mask, out=mask)
    np.logical_not(mask, out=mask)
    np.maximum(merged_data, new_data, out=merged_data, where=mask)
    np.logical_not(new_mask, out=mask)
    np.logical_and(merged_mask, mask, out=mask)
    np.copyto(merged_data, new_data, where=mask, casting="unsafe")

MERGE_METHODS = {
    'first': copy_first,
    'last': copy_last,
    'min': copy_min,
    'max': copy_max
}

def get_tile(shpfile, filename):
    # filename should be m_*tif
    assert filename[0] == 'm', "filename should be of form m_*tif"
    assert filename[-1] == 'f', "filename should be of form m_*tif"
    filename = filename.split('/')[0]
    names = shpfile.FileName.tolist()
    names = [f"{n.rsplit('_', 1)[0]}.tif"for n in names]
    ind = names.index(filename)
    return shpfile.iloc[ind]

def convert_points(lats, lons, src_crs, dest_crs, dest_trans):
    if not isinstance(lats, list):
        lats = [lats]
        lons = [lons]
    if not isinstance(dest_crs, str):
        dest_crs = dest_crs.to_string()
    if not isinstance(src_crs, str):
        src_crs = src_crs.to_string()
    crx, cry = fiona.transform.transform(src_crs, dest_crs, lons, lats)
    memes = [~dest_trans * (x, y) for x, y in zip(crx, cry)]
    return memes


def Find_Rasters_Point(gdf : NAIP_shpfile, point : Point,  base_dir  : str,ftype : str = 'tif'):
    #check_bands(bands)
    rasters = gdf[gdf.contains(point)]
    rasters = [f"{base_dir}/{fman.APFONAME[:5]}/{'_'.join(fman.FileName.split('_')[:-1])}.{ftype}" for _, fman in rasters.iterrows()]
    return rasters
    # can consider returning to opening the raster
# TODO: the boundaries stored in the shp files aren't super accurate - don't use them!
# Turns out thery are, we should use them!
def Find_Boundary_Point(gdf : NAIP_shpfile, point : Point,  base_dir  : str):
    rasters = gdf[gdf.contains(point)]
    return rasters.geometry
    # can consider returning to opening the raster

def Find_Boundary_Polygon(gdf : NAIP_shpfile, poly : Polygon, base_dir : str):
    null = Polygon()
    tt = gdf.intersection(poly)
    rasters =  gdf[tt != null]
    return rasters.geometry

def cast_crs(fname, save_append, dest_crs):
    file = rasterio.open(fname)
    transform, width, height = rasterio.warp.calculate_default_transform(file.crs, dest_crs, file.width, file.height, *file.bounds)
    destination = np.zeros([len(file.indexes), height, width], file.profile['dtype']) # try flipped
    warped, new_affine = rasterio.warp.reproject(file.read(),destination, src_transform=file.transform,
                src_crs=file.crs,
                dst_transform=transform,
                dst_crs=dest_crs,
                resampling=Resampling.nearest)
    # finally resave new version
    f = f"{fname.rsplit('/', 1)[0]}/{save_append}_{fname.rsplit('/')[-1]}"
#     print(f)
    out_profile = file.profile
    out_profile['transform'] = new_affine
    out_profile['height'] = height
    out_profile['width'] = width
    out_profile['count'] = len(file.indexes)
    out_profile['crs'] = dest_crs
#     print("saving file now")
    with rasterio.open(f, 'w', **out_profile) as dst:
        dst.write(warped)
    
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
    method="first", # TODO: add a method that covers up nan values ideally, that's likely where the seaming is coming from
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
        first_crs = first.crs
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
    if bounds is not None:
        dst_w, dst_s, dst_e, dst_n = bounds
    else:
        # scan input files
        xs = []
        ys = []
        reses = []
        for dataset in datasets:
            with dataset_opener(dataset) as src:
                if src.crs != first_crs:
                    raise ValueError(f"raster CRS {src.crs} does not equal starter CRS! {first_crs}")
#                 if src.res != first_res: #TODO: come up with a solution of what to do when resolutions differ...
#                     raise ValueError(f"Resolutions of rasters dont match! {src.res} vs. {first_res}")
                left, bottom, right, top = src.bounds
            xs.extend([left, right])
            ys.extend([bottom, top])
            reses.append(src.res)
        dst_w, dst_s, dst_e, dst_n = min(xs), min(ys), max(xs), max(ys)

    # Resolution/pixel size
    if not res:
        res = first_res
    elif not np.iterable(res):
        res = (res, res)
    elif len(res) == 1:
        res = (res[0], res[0])
        

# TODO: right now the code for calculating the transform and the bounds assumes
# that the resolutions are the same across tiffs, which is not true. Need to correctly
# calculate this...
    # print(f"resolution is {res}, reses are {reses}")
    if target_aligned_pixels:
        dst_w = math.floor(dst_w / res[0]) * res[0]
        dst_e = math.ceil(dst_e / res[0]) * res[0]
        dst_s = math.floor(dst_s / res[1]) * res[1]
        dst_n = math.ceil(dst_n / res[1]) * res[1]
    print(f"res: {res} {dst_w} {dst_e}. {dst_s} {dst_n}")
    # Compute output array shape. We guarantee it will cover the output
    # bounds completely
# because round() has weird behavior, instead we're going to
# round up the size of the array always and slightly
# stretch the rasters to fill it
# TODOO: resolution is not quite right here, because some rasters have a different resolution.
    output_width = math.ceil((dst_e - dst_w) / res[0])
    output_height = math.ceil((dst_n - dst_s) / res[1])
    output_transform = Affine.translation(dst_w, dst_n) * Affine.scale(res[0], -res[1])
# get the bounding box and height / width in order to do future operations
    left = dst_w
    bottom = dst_s
    right = dst_e
    top = dst_n
    d_bounds = rasterio.coords.BoundingBox(left, bottom, right, top)

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
            crs = src.crs

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
        return dest, output_transform, d_bounds, output_width, output_height, crs

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


    # new version
def Mask_Raster(raster : np.array, trans, polygon, crs=NAIP_CRS, dest=None, nodata=None, crop=True, invert=False, pad=False):
    kwargs = {
        'transform' :trans,
        'driver': 'GTiff',
        'height' : raster.shape[1],
        'width' : raster.shape[2],
        'dtype' : rasterio.float64,
        'count' : raster.shape[0],
        'crs' : crs,
        'nodata' : nodata
    }

    # TODO: check out what blockxsize and blockysize do and if they need to be updated also
    with MemoryFile() as memfile:
        with memfile.open(**DefaultGTiffProfile(**kwargs)) as dataset: # Open as DatasetWriter
            profile = dataset.profile
            profile.update(**kwargs)
            # TODO: what if it's a different dtype?
            dataset.write(raster.astype(rasterio.float64))
        with memfile.open(transform=trans, dtype=rasterio.float64) as dataset:  # Reopen as DatasetReader
            # this nodata check isn't doing anything? TODO:
            if nodata is None:
                cropped, trans = mask(dataset, polygon.geometry, crop=crop, nodata=nodata, invert=invert, pad=pad)
            else:
                cropped, trans = mask(dataset, polygon.geometry, crop=crop, nodata=nodata, invert=invert, pad=pad)
            if dest is not None: 
                prof = dataset.profile
                prof['transform'] = trans
                with rasterio.open(dest, 'w', **prof) as dst:
                    dst.write(cropped)
            return cropped, trans
        
def Grab_TIFF(df_row, tif_dir):
        return f"{tif_dir}{df_row.APFONAME[:5]}/{'_'.join(df_row.FileName.split('_')[:-1])}.tif"

# fname: absolute path minus number and filetype
def rename_vrt(fname, n_img,  tot_col, tot_row, vrt_col, vrt_row, ftype='.tif', savetype='.pdf', del_old=True):
      wid = math.ceil(tot_col / vrt_col)
      hid = math.ceil(tot_row / vrt_row)
      collabs = list(string.ascii_uppercase)
      collabs += [f"{lab}{lab}" for lab in collabs]
      print(collabs)
      rowlabs = list(range(1, hid + 1))
      dpi = 200
      # assume that files go from 0-n
 # TODO: fix this: RuntimeWarning: More than 20 figures have been opened. Figures created through the pyplot interface          (`matplotlib.pyplot.figure`) are retained until explicitly closed and may consume too much memory. (To control this warning,   see the rcParam `figure.max_open_warning`).
      for i in range(n_img):
          fig, ax = plt.subplots(figsize=(15,15), dpi=dpi)
          ax.axis("off")
          file = f"{fname}{i}{ftype}"
          with rasterio.open(file) as r:
              img = r.read([1,2,3])
              row = math.floor(i/wid)
              col = i%wid
              name = f"{collabs[col]}{rowlabs[row]}"
              ax.imshow(np.rollaxis(img, 0, 3))
              ax.text(30, 100, name, color="w", fontsize=30)
              fnamm = f"{fname.rsplit('.', 2)[0]}_{name}{savetype}"
              fig.savefig(fnamm, dpi=dpi)
          if del_old:
              os.remove(file)

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



def predict_raster_list(device_no, tiffs, modelname, res, year, means, model_pth, cfg_pth, base_dir, warp):
    # TODO: remove this garbage belwo
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
        if warp and src.crs != NAIP_CRS:
            # use rasterio to reproject https://rasterio.readthedocs.io/en/latest/topics/reproject.html
            nnt, wid, hig = calculate_default_transform(src.crs, NAIP_CRS, src.width, src.height, *src.bounds)
            kwargs.update({
                'width': wid,
                'height': hig,
                'transform' : nnt,
                'crs' : NAIP_CRS,
            })
            dest = np.full([hig, wid], nodata, dtype=dtype)
            reproject(output, dest, src_transform=src.transform, src_crs=src.crs, dst_transform=nnt,dst_crs=NAIP_CRS,resampling=Resampling.bilinear)
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


def predict_raster(rasname, model, batchsize, res, year, base_dir, modelname, num_spec, device, spec_names, warp, means, savename=None):
    with rasterio.open(rasname) as src:
        ras = src.read()
    # so if the model is an old model, need to modify the input to be scaled
    # do it here so that edge cases dont accidentally subtract twice
        if "old" in modelname:
            for i, (channel, mean) in enumerate(zip(means, ras)):
                ras[i,:,:] = mean - channel
                
        else:
            dat = utils.scale(dat, out_range=(0,1), min_=0, max_=255)
            datt = np.copy(dat)
            datt = datt.astype(np.float)
            for channel in range(len(means)):
                datt[channel,:,:] = (dat[channel,:,:]-means[channel])/std[channel]
        width, height = src.width, src.height
        num_w, num_h  = width//res, height//res
        clean_w, clean_h = num_w *res, num_h*res
# TODO: is this + 1 the reason there's seams??
        new_w = num_w + 1 if width % res != 0 else num_w
        new_h = num_h + 1 if height % res != 0 else num_h
        output = np.zeros([num_spec, new_h, new_w]) 
        images = []
        # grab all the locations that normally fit
        for i in range(0, clean_w, res):
            for j in range(0, clean_h, res):
#                 print(f" j {j}, i {i}, res {res}")
                ii, jj = int(i/res), int(j/res)
                window = ras[:, j:j+res, i:i+res]
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
        kwargs = src.meta.copy()
        output = np.float32(output)
        if warp:
                # use rasterio to reproject https://rasterio.readthedocs.io/en/latest/topics/reproject.html
            # there's a chance that there's some tomfoolery in the order of
            # bounds and w/h, but I'm going to trust that it works for now
            nt = rasterio.transform.from_bounds(*src.bounds, new_w, new_h)
            nnt, wid, hig = calculate_default_transform(src.crs, NAIP_CRS, new_w, new_h, *src.bounds)
            kwargs.update({
                'width': wid,
                'height': hig,
                'count' : num_spec,
                'transform' : nnt,
                'crs' : NAIP_CRS,
                'dtype' : 'float32'
                # so I ran torchy sig with both 64 and 32 and nothing seemed to change so ignoring for now
            })
            dest = np.zeros([num_spec, hig, wid])
            reproject(output, dest, src_transform=nt, src_crs=src.crs, dst_transform=nnt,dst_crs=NAIP_CRS,resampling=Resampling.bilinear)
            output = dest
        else:
            nt = rasterio.transform.from_bounds(*src.bounds, new_w, new_h)
            kwargs.update({
                'width': new_w,
                'height': new_h,
                'count' : num_spec,
                'transform' : nt,
                'dtype' : 'float32'
                # so I ran torchy sig with both 64 and 32 and nothing seemed to change so ignoring for now
            })
            if savename is None:
                # TODO: this is broken, specifically the split part
                fname = f"{base_dir}inference/prediction/raw/{modelname}/{rasname.split(paths.NAIP_BASE)[-1]}" # and steal filename from NAIP
            else:
                fname = savename
            if not os.path.exists(fname.rsplit('/',1)[0]): # make directory if needed
                os.makedirs(fname.rsplit('/',1)[0])
            with rasterio.open(fname, 'w', **kwargs) as dst:
                dst.write(output, range(1,len(spec_names)+1))
                dst.descriptions = spec_names
    return fname

def predict_raster_arbitrary_res(sat_file, save_file, b_size, res, spec_names, device, model, modelname, means, std=0, img_size=256):
    # TODO: if batch size is too big, this breaks??
    tock = time.time()
    with rasterio.open(sat_file) as sat:
        dat = sat.read()
        swidth, sheight = sat.width, sat.height
        if b_size > swidth:
            raise NotImplementedError
        # leave off the last pixels for which we don't have a full 256 image for
        # TODO: remove magic number ofi mage size
        i_ind = range(0, swidth-img_size, res)
        j_ind = range(0, sheight-img_size, res)
        n_specs = len(spec_names)
             #TODO: hacky, eventually change or remove
        if "old" in modelname:
            # the order is messed up but it's that way for all other 
            # network predictions, including how it was trained
            # so oh well...
            for i, (channel, mean) in enumerate(zip(means, dat)):
                dat[i,:,:] = mean - channel
        else:
            dat = utils.scale(dat, out_range=(0,1), min_=0, max_=255)
            datt = np.copy(dat)
            datt = datt.astype(np.float)
            for channel in range(len(means)):
                datt[channel,:,:] = (dat[channel,:,:]-means[channel])/std[channel]
            dat = datt
        wid, hig = len(i_ind), len(j_ind)
        # make receiver array
        result = np.full([n_specs, hig, wid],np.nan,  dtype=np.float64)
        ii = 0

        with torch.no_grad():
            if b_size > hig:
                with tqdm(total=(math.ceil(hig/(b_size//hig))), unit="window") as prog:
                    # if it can handle huge batch sizes, then take the 
                    # floor(# rows this thing can take)
                    # going to forget the leftover bit for now too much effort for not that much speedup
                    for i in utils.chunks(i_ind, (b_size//hig)):
                        ba  = [[dat[:, c:c+img_size, r:r+img_size] for c in j_ind] for r in i]
                        ba = np.vstack(ba)
                        tc = torch.tensor(ba, dtype=torch.float)
                        tc = tc.to(device)
                        out, a, b = model(tc)
                        out = np.squeeze(out.detach().cpu().numpy())
                        for k in range(len(i)):
                            result[:,:,ii] = out.T[:,k:k+hig]
                            ii += 1
                        prog.update(1)

            else:
                with tqdm(total=(len(i_ind)*(len(j_ind)//b_size)), unit="window") as prog:
                    for i in i_ind:
                        jj = 0
                        for j in utils.chunks(j_ind, b_size):
                            # max batch size this way is len(j_ind)...
                            ba = [dat[:, c:c+img_size, i:i+img_size] for c in j]
                            # it really is the size lol
                            tc = torch.tensor(ba, dtype=torch.float)
                            tc = tc.to(device)
                            out, a, b = model(tc)
                             #  TODO: make sure flipping i, j fixed problems
                            out = np.squeeze(out.detach().cpu().numpy())
                            # TODO: this will fail on the last row, likely will need to troubleshoot the leftovers
                            # doesn't fail b/c we cut off the leftovers above
                            result[:,jj:jj+b_size,ii] = out.T
                            jj +=b_size
                            prog.update(1)
                        ii += 1
            prog.close()


        out_profile = sat.profile
        bounds = sat.bounds
    out_res = (sat.width/wid, sat.height/hig)
    out_profile['res'] = out_res
    # dst_w = left, dst_n = top
    trans = Affine.translation(bounds.left, bounds.top) * Affine.scale(out_res[0], -out_res[1])
    out_profile['transform'] = trans
    out_profile['height'] = hig
    out_profile['width'] = wid # +1 ecause there's the leftover bits? or is it -1 on the range?
    out_profile['count'] = n_specs
    out_profile['dtype'] = np.float64
    out_profile.update(BIGTIFF="IF_SAFER")
    print("saving file now")
    
    with rasterio.open(save_file, 'w', **out_profile) as dst:
        dst.write(result, range(1,n_specs+1))
        dst.descriptions = spec_names
    tick = time.time()
    print(f"file {save_file.split('/')[-1]} took {(tick-tock)/60} minutes")
    
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
        # batch size of 5 didn't work, try incrementally smaller
        # TODO: make this cleaner, modular
    for i in range(5, 1, -1):
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
    if good:
        print(f"batch size is {i}")
        return i
