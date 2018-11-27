#!/usr/bin/env python

import ee
import ee.mapclient
ee.Initialize()
import urllib.request
# import geopandas as gpd
import requests
import zipfile
import io
import os
import subprocess

from DEEPBIO_EE import *
#################################################################################
## Query server for asset names


foldergee='users/moisesexpositoalonso'
p = subprocess.Popen(["earthengine", "ls",foldergee], stdout=subprocess.PIPE)
out, err = p.communicate()
assets=out.decode("utf-8").split("\n")

#imgnames= list(filter(lambda x:'122' in x, assets)) # for test with 1 image
imgnames= list(filter(lambda x:'36' in x, assets)) # a latitudinal band
basenames=[os.path.basename(i) for i in imgnames]


for path,base in zip(imgnames,basenames):
    outpath=os.path.join('../satellite',base)
    #################################################################################
    # download
    print('Requesting url for user image: %s ...'%(path))
    image=ee.Image(path)
    url=image.getDownloadUrl({})
    #################################################################################
    # Get a download URL for an image.
    print('Downloading zip image from: %s'%(url))
    r = requests.get(url)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    print("Extracting zip to: %s" %(outpath))
    z.extractall(path=outpath) # extract to folder
    #################################################################################
    # read images
    ima=readsatelliteimages(outpath)
    #fil=open(os.path.join(outpath,"".join([base,".B10.tfw"]))).read() 
    np.save(os.path.join(outpath,"".join([base,".npy"])), ima)
    
