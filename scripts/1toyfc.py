import pandas as pd
import os
import matplotlib.pyplot as plt 
#import ee
#import ee.mapclient
#ee.Initialize()
from PIL import Image
import numpy as np

#################################################################################
## Import image
lon='36.6'
lat='-122'
step=1

fi='../sat/1deg_36dot6_-122.bio07.tif'
im=Image.open(fi)
ima=np.array(im)

print(ima)

###########################################################################################
# read gbif dataset
d = pd.read_csv("../gbif/0002455-181108115102211.csv")
d.head()
print(d.size)
