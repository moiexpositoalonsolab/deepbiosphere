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

lat, lon = np.array(parsed,dtype='f').min(axis=0)
lat1, lon1 = np.array(parsed,dtype='f').max(axis=0)


################################################################################
### Read gbif dataset and make the label tensor
################################################################################
from DEEPBIO_GBIF import *
from UTILS import *
print("Loading GBIF data")

pathgbif="../gbif"
# filegbif="pgbif.csv" # this is family
filegbif="fgbif.csv" # this is for species
filegbif="ggbif.csv" # this is for species
filegbif="sgbif.csv" # this is for species

# read gbif dataset
d=readgbif(path=os.path.join(pathgbif,filegbif))
dg=readgbif(path=os.path.join(pathgbif,"ggbif.csv"))
df=readgbif(path=os.path.join(pathgbif,"fgbif.csv"))

# this is to reduce the number of classes
d_ = subcoorgrid(d,lat,lat1,lon,lon1)
abundances=d_['species'].value_counts() # important for species, very slow otherwise
enough = abundances[abundances>=500]
keep=d_["species"].isin(list(enough.index))
d_ =d_[keep]
d=d_

dg=subcoorgrid(dg,lat,lat1,lon,lon1)[keep]
df=subcoorgrid(df,lat,lat1,lon,lon1)[keep]

# make spp hash
spp=makespphash(d.iloc[:,0])
sppdic=make_sppdic(spp,len(spp))

gen=makespphash(dg.iloc[:,0])
gendic=make_sppdic(gen,len(gen))

fam=makespphash(df.iloc[:,0])
famdic=make_sppdic(fam,len(fam))

# make species map
print("Generating biodiversity map")
sampledensity,spptensor= vec_tensorgbif(parsed,step,breaks,d,sppdic,'yesno')
sampledensity,gtensor= vec_tensorgbif(parsed,step,breaks,dg,gendic,'yesno')
sampledensity,ftensor= vec_tensorgbif(parsed,step,breaks,df,famdic,'yesno')


np.save(os.path.join(pathgbif,"gbiftensorspp.npy"), np.array(spptensor))
np.save(os.path.join(pathgbif,"gbiftensorg.npy"), np.array(gtensor))
np.save(os.path.join(pathgbif,"gbiftensorf.npy"), np.array(ftensor) )

np.save(os.path.join(pathgbif,'gbifdicspp.npy'), sppdic)
np.save(os.path.join(pathgbif,'gbifdicg.npy'), gendic)
np.save(os.path.join(pathgbif,'gbifdicf.npy'), famdic)

## save the grid
#npspptensor=np.array(spptensor)
#np.save(os.path.join(pathgbif,"gbiftensor.npy"), npspptensor)
#npsampledensity=np.array(sampledensity)
#np.save(os.path.join(pathgbif,"gbifdensity.npy"), npsampledensity)
#np.save(os.path.join(pathgbif,'gbifdic.npy'), sppdic)

# output also info of the order of layers
f=open(os.path.join(pathgbif,"gbiftensor.info"), "w")
for n,i in zip(cells,parsed):
#    f.write("{}\t{}\t{}\n".format(n,i[0],i[1]))
    f.write("{}\n".format(n))

f.close()


print("Done")
