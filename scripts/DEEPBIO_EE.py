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
    im=np.nan_to_num(im)
    im[im < (-1 * precfloat)] =0
    im[im > (precfloat)] =0 # better brute force it
    return(im)

def readsatelliteimages(path):
    files=findtiffiles(path)
    r=[readtif2np(fi) for fi in files]
    return(np.array(r))

def subsetimagetensor(ima,y,x,z,linearize='cnn',pixside=50,breaks=10):
    wind=[[pixside*i,(pixside)+pixside*i] for i in range(int(breaks))]
    inputs=[ ima[l, : , wind[i][0]:wind[i][1]  ,  wind[j][0]:wind[j][1] ]  for l,i,j in zip(z,y,x)] # the [] important to define dymensions
    inputs=np.array(inputs, dtype='f')
    inputs=torch.from_numpy(inputs)
    if(args.nn=="fc"):
        inputs=inputs.view(-1, par.numchannels*par.pixside*par.pixside) # this is for fully connected
    return(inputs)



