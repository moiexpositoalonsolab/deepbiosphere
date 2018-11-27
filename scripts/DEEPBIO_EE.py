"""
Deployment of Google Earth Engine API
@author: moisesexpositoalonso@gmail.com

"""
# import geopandas as gpd
# import rasterio
# from rasterio import features

## Utilities
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import numpy as np
import math

## To work with Earth Engine
import ee
import ee.mapclient
ee.Initialize()

precfloat=1e38

################################################################################
## Utilities
def readraster(file="tmp/SRTM90_V4.elevation.tif"):
    rst= rasterio.open(file)
    return(rst)

def readimgnp(file='../sat/exampleExport01deg.B1.tif'):
    im=Image.open(fi)
    ima=np.array(im)
    return(ima)

def findtiffiles(path, ext='.tif'):
    tifs=[]
    for file in os.listdir(path):
        if file.endswith(ext):
            tifs.append(join(path,file))
    return(tifs)

def readtif2np(fi):
    im=Image.open(fi)
    im=np.array(im)
    im[im < (-1 * precfloat)] =0
    im[im > (precfloat)] =0 # better brute force it
    im = (im - im.mean()) / (math.sqrt(im.var()))
    return(im)

def readsatelliteimages(path):
    files=findtiffiles(path)
    r=[readtif2np(fi) for fi in files]
    return(np.array(r))
