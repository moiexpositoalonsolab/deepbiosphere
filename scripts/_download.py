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

###############################################################################
## define names

#lon=-122
#lat=36.6
#imgname='users/moisesexpositoalonso/exampleExport'
#imgname='users/moisesexpositoalonso/exampleExport01deg'
foldergee='users/moisesexpositoalonso/'
imgname='1deg_36dot6_-122'
path=os.path.join(foldergee,imgname)
print('Requesting path for user image: %s ...'%(path))

outpath='../satellite'
#################################################################################
# download

image=ee.Image(path)
#path = image.getDownloadUrl({
#    'scale': 10,
#    'crs': 'EPSG:4326',
# #  'region': '[[-122, 36.6], [-122, 35], [-119, 36.6], [-119, 35]]',
#    'maxPixels': '1e6'
#})
url=image.getDownloadUrl({})

#################################################################################


# Get a download URL for an image.
print('Downloading zip image from: %s'%(url))
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))

print("Extracting zip to: %s" %(outpath))
z.extractall(path=outpath) # extract to folder




