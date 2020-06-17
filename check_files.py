import pandas as pd
import os
from os import listdir
import pprint
import argparse
import numpy as np
import matplotlib.pyplot as plt
from deepbiosphere.scripts import paths
import math
import glob
import os, os.path



def main():
    pth = ARGS.base_dir 
    us_train_pth = "{}occurrences/occurrences_us_train.csv".format(pth)
    us_train = pd.read_csv(us_train_pth, sep=';')
    us_test_pth = "{}occurrences/occurrences_us_test.csv".format(pth)
    us_test = pd.read_csv(us_test_pth, sep=';')#
    fr_train_pth = "{}occurrences/occurrences_fr_train.csv".format(pth)
    fr_train = pd.read_csv(fr_train_pth, sep=';')
    fr_test_pth = "{}occurrences/occurrences_fr_test.csv".format(pth)
    fr_test = pd.read_csv(fr_test_pth, sep=';')#
    # grab all the image files (careful, really slow!)
    us_dir = "({}patches_us)".format(pth)
    #for root, dirs, files in os.walk('python/Lib/email'):
    #print len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    paths_us = glob.glob("{}/patches_us/patches_us_*/*/*/*_alti.npy".format(pth))
    paths_fr = glob.glob("{}/patches_fr/*/*/*_alti.npy".format(pth))

    # how many images are missing?
    us_summed = us_train.shape[0] + us_test.shape[0] 
    fr_summed = fr_train.shape[0]  +fr_test.shape[0]
    us_missed = us_summed - len(paths_us)
    fr_missed = fr_summed - len(paths_fr)
    print("number of us missing files: {} number of missing fr files: {}".format(us_missed, fr_missed))

    # get ids of files that are present
    us_path_ids = [ path.split("_alti")[0].split("/")[-1] for path in us_paths]

    fr_path_ids = [ path.split("_alti")[0].split("/")[-1] for path in fr_paths]
    # grab ids from both train and test set
    us_cat_ids = pd.concat([us_train['id'], us_test['id']])
    fr_cat_ids = pd.concat([fr_train['id'], fr_test['id']])

    # get the ids that are missing from the image dataset
    us_missing = us_cat_ids[~us_cat_ids.isin(us_path_ids)]
    fr_missing = fr_cat_ids[~fr_cat_ids.isin(fr_path_ids)]

    # build a set of all the directories that are missing in the data
    missing_folders = set()
    for miss in us_missing:
        abcd = miss % 10000
        ab, cd = math.floor(abcd/100), abcd%100
        cdd = math.ceil((cd+ 1)/5)
        cdd = "0{}".format(cdd)  if cdd < 10 else "{}".format(cdd)
        ab = "0{}".format(ab) if miss / 1000 > 1 and ab < 10 else ab
        cd = "0{}".format(cd) if miss / 1000 > 1 and cd < 10 else cd
        subpath = "patches_us_{}/{}/{}/".format(cdd, cd, ab)
        missing_folders.add(subpath)
    pprint.pprint("missing us folders: {}".format(missing_folders))
    missing_fr_folders = set()
    for miss in fr_missing:
        abcd = id_ % 10000
        ab, cd = math.floor(abcd/100), abcd%100
        ab = "0{}".format(ab) if id_ / 1000 > 1 and ab < 10 else ab
        cd = "0{}".format(cd) if  cd < 10 else cd
        subpath = "patches_{}/{}/{}/".format(country, cd, ab)
        missing_fr_folders.add(subpath)
    pprint.pprint("missing fr folders: {}".format(missing_fr_folders))

if __name__ == "__main__":
    #print(f"torch version: {torch.__version__}") 
    #print(f"numpy version: {np.__version__}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", type=str, help="what folder to read images from",choices=['DBS_DIR', 'MEMEX_LUSTRE', 'CALC_SCRATCH', 'AZURE_DIR'], required=True)
    ARGS, _ = parser.parse_known_args()
    # parsing which path to use
    ARGS.base_dir = eval("paths.{}".format(ARGS.base_dir))
    print("using base directory {}".format(ARGS.base_dir))
    main()
