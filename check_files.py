import pandas as pd
import os
from os import listdir
import pprint
import numpy as np
import matplotlib.pyplot as plt
from deepbiosphere.scripts import paths
import math
import os, os.path



def main():
    pth = ARGS.path 
    us_train_pth = f"{pth}occurrences/occurrences_us_train.csv"
    us_train = pd.read_csv(us_train_pth, sep=';')
    us_test_pth = f"{pth}occurrences/occurrences_us_test.csv"
    us_test = pd.read_csv(us_test_pth, sep=';')#
    fr_train_pth = f"{pth}occurrences/occurrences_fr_train.csv"
    fr_train = pd.read_csv(fr_train_pth, sep=';')
    fr_test_pth = f"{pth}occurrences/occurrences_fr_test.csv"
    fr_test = pd.read_csv(fr_test_pth, sep=';')#
    # grab all the image files (careful, really slow!)
    us_dir = f"{pth}patches_us}"
    #for root, dirs, files in os.walk('python/Lib/email'):
    #print len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    paths_us = glob.glob(f"{pth}/patches_us/patches_us_*/*/*/*_alti.npy")
    paths_fr = glob.glob(f"{pth}/patches_fr/*/*/*_alti.npy")

    # how many images are missing?
    us_summed = us_train.shape[0] + us_test.shape[0] 
    fr_summed = fr_train.shape[0]  +fr_test.shape[0]
    us_missed = us_summed - len(paths_us)
    fr_missed = fr_summed - len(paths_fr)
    print(f"number of us missing files: {us_missed} number of missing fr files: {fr_missed}")

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
        cdd = f"0{cdd}"  if cdd < 10 else f"{cdd}"
        ab = f"0{ab}" if miss / 1000 > 1 and ab < 10 else ab
        cd = f"0{cd}" if miss / 1000 > 1 and cd < 10 else cd
        subpath = f"patches_us_{cdd}/{cd}/{ab}/"
        missing_folders.add(subpath)
    pprint.pprint(f"missing us folders: {missing_folders}")
    missing_fr_folders = set()
    for miss in fr_missing:
        abcd = id_ % 10000
        ab, cd = math.floor(abcd/100), abcd%100
        ab = "0{}".format(ab) if id_ / 1000 > 1 and ab < 10 else ab
        cd = "0{}".format(cd) if  cd < 10 else cd
        subpath = "patches_{}/{}/{}/".format(country, cd, ab)
        missing_fr_folders.add(subpath)
    pprint.pprint(f"missing fr folders: {missing_fr_folders}")

if __name__ == "__main__":
    #print(f"torch version: {torch.__version__}") 
    #print(f"numpy version: {np.__version__}")
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, help="learning rate of model",required=True)
    parser.add_argument("--epoch", type=int, required=True, help="how many epochs to train the model")
    parser.add_argument("--device", type=int, help="which gpu to send model to, don't put anything to use cpu")
    parser.add_argument("--processes", type=int, help="how many worker processes to use for data loading", default=1)
    parser.add_argument("--exp_id", type=str, help="experiment id of this run", required=True)
    parser.add_argument("--base_dir", type=str, help="what folder to read images from",choices=['DBS_DIR', 'MEMEX_LUSTRE', 'CALC_SCRATCH', 'AZURE_DIR'], required=True)
    parser.add_argument("--country", type=str, help="which country's images to read", default='us', required=True, choices=['us', 'fr', 'both'])
    parser.add_argument("--seed", type=int, help="random seed to use")
    parser.add_argument('--test', dest='test', help="if set, split train into test, val set. If not seif set, split train into test, val set. If not set, train network on full dataset", action='store_true')
    parser.add_argument("--load_size", type=int, help="how many instances to hold in memory at a time", default=1000)
    parser.add_argument("--batch_size", type=int, help="size of batches to use", default=50)
    ARGS, _ = parser.parse_known_args()
    # parsing which path to use
    ARGS.base_dir = eval("paths.{}".format(ARGS.base_dir))
    print("using base directory {}".format(ARGS.base_dir))
    # Seed
    assert ARGS.load_size >= ARGS.batch_size, "load size must be bigger than batch size!"
    if ARGS.seed is not None:
        np.random.seed(ARGS.seed)
        torch.manual_seed(ARGS.seed)
    if not os.path.exists("{}output/".format(ARGS.base_dir)):
        os.makedirs("{}output/".format(ARGS.base_dir))
    if not os.path.exists("{}nets/".format(ARGS.base_dir)):
        os.makedirs("{}nets/".format(ARGS.base_dir))
    main()
