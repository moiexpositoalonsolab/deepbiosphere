#!/usr/bin/env python
"""
Parse GBIF and construct tensor
@author: moisesexpositoalonso@gmail.com

"""


import pandas as pd
import os
from os import listdir
from os.path import isfile, join

from PIL import Image
import numpy as np


# Seed
seed = 1
np.random.seed(seed)


# Analysis design
#pixside=50
pixside=100
breaks=int(500/pixside)
step=1

################################################################################
## List folders of images
################################################################################
print("Finding locations for satellite images")
path="../satellite"

cells=[os.path.basename(os.path.join(path, o)) for o in os.listdir(path)
        if os.path.isdir(os.path.join(path,o))]
cells.sort()
parsed=[i.replace("1deg_","").replace("dot",".").split("_") for i in cells]

################################################################################
### Read gbif dataset and make the label tensor
################################################################################
from DEEPBIO_GBIF import *
print("Loading GBIF data")

pathgbif="../gbif"
filegbif="pgbif.csv"

# read gbif dataset
d=readgbif(path=os.path.join(pathgbif,filegbif))
spp=makespphash(d.iloc[:,0])
spptot=len(spp)
sppdic=make_sppdic(spp,spptot)

# make species map
print("Generating biodiversity map")
sampledensity,spptensor= vec_tensorgbif(parsed,step,breaks,d,sppdic,'yesno')
# sampledensity,spptensor= vec_tensorgbif(parsed[0:2],step,breaks,d,sppdic,'yesno')
#sampledensity,biogrid=[tensorgbif(step, breaks, float(latlon[0]),float(latlon[1]), d, sppdic,'yesno') for latlon  in parsed[0:2] ]
#biogrid= [tensoronetaxon(step, breaks, float(latlon[0]),float(latlon[1]), d, 'Poaceae') for latlon  in parsed ]
#biogrid= [tensoronetaxon(step, breaks, float(latlon[0]),float(latlon[1]), d, 'Pinaceae') for latlon  in parsed ]
# biogrid=tensorgbif(step, breaks, lat,lon, d) ## for many classes
#print(biogrid)

# save the grid
npspptensor=np.array(spptensor)
np.save(os.path.join(pathgbif,"gbiftensor.npy"), npspptensor)
npsampledensity=np.array(sampledensity)
np.save(os.path.join(pathgbif,"gbifdensity.npy"), npsampledensity)
np.save(os.path.join(pathgbif,'gbifdic.npy'), sppdic)

# output also info of the order of layers
f=open(os.path.join(pathgbif,"gbiftensor.info"), "w")
for n,i in zip(cells,parsed):
#    f.write("{}\t{}\t{}\n".format(n,i[0],i[1]))
    f.write("{}\n".format(n))

f.close()


print("Done")
