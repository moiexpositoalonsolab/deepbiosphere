"""
Download user assets from Google Earth Engine
@author: moisesexpositoalonso@gmail.com

"""

import urllib.request
import requests
import zipfile
import io
import os
import subprocess

import ee
import ee.mapclient
ee.Initialize()

from DEEPBIO_EE import *

force=False
#################################################################################
## Query server for asset names


foldergee='users/moisesexpositoalonso'
p = subprocess.Popen(["earthengine", "ls",foldergee], stdout=subprocess.PIPE)
out, err = p.communicate()
assets=out.decode("utf-8").split("\n")

#imgnames= list(filter(lambda x:'122' in x, assets)) # for test with 1 image
#imgnames= list(filter(lambda x:'36' in x, assets)) # a latitudinal band
imgnames=assets
imgnames.sort() # a latitudinal band
basenames=[os.path.basename(i) for i in imgnames]
basenames

imgnames = list(filter(None, imgnames))
basenames = list(filter(None, basenames))

satellitepath="../satellite"

raster=[]
#path,base = list(zip(imgnames,basenames))[0]
for path,base in zip(imgnames,basenames):
    outpath=os.path.join(satellitepath,base)
    ################################################################################
    if(os.path.exists(outpath) and not force):
        print("Image folder already found: %s" %(outpath))
    else:
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
        ################################################################################
    # read images
    ima=readsatelliteimages(outpath)
    print(ima.shape)
    ima=ima[:,0:500,0:500]
    print("Loaded all images")
    ####fil=open(os.path.join(outpath,"".join([base,".B10.tfw"]))).read()# this can be used to get image information
    #save
    np.save(os.path.join(outpath,"".join([base,".npy"])), ima)
    raster.append(ima)


# save
ima=np.array(raster)
np.save(os.path.join(satellitepath,"".join(["rastersraw",".npy"])), ima)
print("Normalizing all images")
for i in range(ima.shape[1]): # manual normalization per layer
    ima[:,i,:,:] =  (ima[:,i,:,:] -  ima[:,i,:,:].mean())  / ima[:,i,:,:].var()
np.save(os.path.join(satellitepath,"".join(["rasters",".npy"])), ima)

f=open(os.path.join(satellitepath,"".join(["rasters",".info"])),"w")
for path,base in zip(imgnames,basenames):
    f.write("{}\n".format(base))
f.close()
print("Done")
