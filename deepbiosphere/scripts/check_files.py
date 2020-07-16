import pprint
import argparse
from deepbiosphere.scripts import paths
from deepbiosphere.scripts import GEOCLEF_Config as config
from deepbiosphere.scripts import GEOCLEF_Utils as utils

def main():
    pth = ARGS.base_dir 
    us_train_pth = "{}occurrences/occurrences_us_train.csv".format(pth)
    us_test_pth = "{}occurrences/occurrences_us_test.csv".format(pth)    
    paths_us = "{}/patches_us/patches_us_*/*/*/*_alti.npy".format(pth)
    missing_us_folders = utils.check_gbif_files([us_train_pth, us_test_pth], paths_us)
    pprint.pprint(missing_us_folders)
    
    fr_train_pth = "{}occurrences/occurrences_fr_train.csv".format(pth)
    fr_test_pth = "{}occurrences/occurrences_fr_test.csv".format(pth)
    paths_fr = "{}/patches_fr/*/*/*_alti.npy".format(pth)
    missing_fr_folders = utils.check_gbif_files([fr_train_pth, fr_test_pth], paths_fr)    
    pprint.pprint(missing_fr_folders)
    




if __name__ == "__main__":
    args = ['base_dir']
    ARGS = config.parse_known_args(args)
    main()
