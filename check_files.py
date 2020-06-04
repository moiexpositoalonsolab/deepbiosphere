import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import pprint
import numpy as np
import matplotlib.pyplot as plt
import glob
from deepbiosphere.scripts import paths
import math
import pprint
import pdb

calc_pth = paths.CALC_SCRATCH
memex_pth = paths.MEMEX_LUSTRE

pth = calc_pth 
us_train_pth = f"{pth}occurrences/occurrences_us_train.csv"
us_train = pd.read_csv(us_train_pth, sep=';')
us_test_pth = f"{pth}occurrences/occurrences_us_test.csv"
us_test = pd.read_csv(us_test_pth, sep=';')#

# grab all the image files (careful, really slow!)
paths = glob.glob(f"{pth}/patches_us/patches_us_*/*/*/*_alti.npy")
len(paths)

# how many images are missing?
summed = us_train.shape[0] + us_test.shape[0] 
missed = summed - len(paths)
print(f"number of missing files: {missed}")

# get ids of files that are present
path_ids = [ path.split("_alti")[0].split("/")[-1] for path in paths]

# grab ids from both train and test set
cat_ids = pd.concat([us_train['id'], us_test['id']])

# get the ids that are missing from the image dataset
missing = cat_ids[~cat_ids.isin(path_ids)]
len(missing)

# build a set of all the directories that are missing in the data
missing_folders = set()
for miss in missing:
    abcd = miss % 10000
    ab, cd = math.floor(abcd/100), abcd%100
    cdd = math.ceil((cd+ 1)/5)
    cdd = f"0{cdd}"  if cdd < 10 else f"{cdd}"
    ab = f"0{ab}" if miss / 1000 > 1 and ab < 10 else ab
    cd = f"0{cd}" if miss / 1000 > 1 and cd < 10 else cd
    subpath = f"patches_us_{cdd}/{cd}/{ab}/"
    missing_folders.add(subpath)

#pprint.pprint(f"missing folders: {missing_folders}")
