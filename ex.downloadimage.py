#!/usr/bin/env python
"""Download example."""

import ee
import ee.mapclient

ee.Initialize()

###############################################################################
## Load layers



#################################################################################
# export

#lon=-122
#lat=36.6
imgname='users/moisesexpositoalonso/exampleExport'
print('Requesting path for user image: %s ...'%(imgname))


image=ee.Image(imgname)
#path = image.getDownloadUrl({
#    'scale': 10,
#    'crs': 'EPSG:4326',
# #  'region': '[[-122, 36.6], [-122, 35], [-119, 36.6], [-119, 35]]',
#    'maxPixels': '1e6'
#})
url=image.getDownloadUrl({})
print (url)

#################################################################################
import PIL # need to install this
from PIL import ImageTk
import ee
import ee.mapclient
import urllib.request
ee.Initialize()
# import geopandas as gpd
import requests
import zipfile
import io


# Get a download URL for an image.
print('Downloading shapefile...')
r = requests.get(url)
z = zipfile.ZipFile(io.BytesIO(r.content))

print("Extracting zip...")
z.extractall(path='tmp/') # extract to folder
# filenames = [y for y in sorted(z.namelist()) for ending in ['dbf', 'prj', 'shp', 'shx'] if y.endswith(ending)]
# print(filenames)



