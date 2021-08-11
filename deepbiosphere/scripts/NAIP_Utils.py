import torch
import matplotlib as mpl
import matplotlib.patches as patches
import time
import math
import matplotlib.cm as cm
import matplotlib as mpl
import time
# deepbio packages
from deepbiosphere.scripts.GEOCLEF_Config import paths
from deepbiosphere.scripts.GEOCLEF_Run import  setup_dataset, setup_model, setup_loss
import deepbiosphere.scripts.GEOCLEF_Dataset as dataset
import deepbiosphere.scripts.GEOCLEF_Config as config
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
from rasterio.windows import Window 
from tqdm import tqdm


import fiona
import fiona.transform
import requests
import json
import torch
import numpy as np


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
        self.tile_rtree = rtree.index.Index(base_path + "/tile_index") # TODO: look and see what these two bad boys contain
        self.tile_index = pickle.load(open(base_path  + "/tiles.p", "rb"))
      
    
    def lookup_tile(self, lat, lon):
        """"
        Given a lat/lon coordinate pair, return the list of NAIP tiles that contain
        that location.

        Returns a list of COG file paths.
        """

        point = shapely.geometry.Point(float(lon),float(lat))
        intersected_indices = list(self.tile_rtree.intersection(point.bounds)) # oh wow so this rtree does ALL the heavy lifting, phew....
        # TODO: time this...
        intersected_files = []
        tile_intersection = False

        for idx in intersected_indices:

            intersected_file = self.tile_index[idx][0]
            intersected_geom = self.tile_index[idx][1] #TODO: figure out what this geom is
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
    
def setup_NAIPIndex(blob_root='https://naipeuwest.blob.core.windows.net/naip', crs='EPSG:4326', index_loc=None):
	
	index_files = ['tile_index.dat', 'tile_index.idx', 'tiles.p'] # TODO: these are the files that tell you which tiles are where
	index_blob_root = re.sub('/naip$','/naip-index/rtree/',blob_root)
	temp_dir = os.path.join(tempfile.gettempdir(),'naip') if index_loc is None else index_loc
	os.makedirs(temp_dir,exist_ok=True)
	# Spatial index that maps lat/lon to NAIP tiles; we'll load this when we first 
	# need to access it.
	index = None
	warnings.filterwarnings("ignore")
	return NAIPTileIndex(index_blob_root, index_files,  temp_dir), crs
