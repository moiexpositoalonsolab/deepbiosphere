from random import randrange
import pandas as pd
import argparse
import numpy as np
import random
import math
from deepbiosphere.scripts import GEOCLEF_CNN as cnn
from deepbiosphere.scripts import GEOCLEF_Dataset as Dataset
from deepbiosphere.scripts.GEOCLEF_Config import paths
import deepbiosphere.scripts.GEOCLEF_Config as config
import deepbiosphere.scripts.GEOCLEF_Utils as utils
import reverse_geocoder as rg

def main():
    print("getting data")
    pth = ARGS.base_dir
    us_train_pth = "{}occurrences/occurrences_us_plantanimal_train.csv".format(pth)
    us_train = pd.read_csv(us_train_pth, sep=';')
    print("filtering by state")
    # create a new tuple column
    us_train['lat_lon'] = list(zip(us_train.lat, us_train.lon))
    # convert to list for faster exraction
    us_latlon = us_train['lat_lon'].tolist()
    # grab location data for the lat lon
    res = rg.search(us_latlon)
    # grab only the states from the results
    states = [r['admin1'] for r in res]
    # add the states information back into the original dataframe
    us_train['state'] = states


    # grab only observations from california
    if ARGS.census:
        filtered_us = Dataset.raster_filter_2_cali(ARGS.base_dir, us_train)
        
    else:
        filtered_us = us_train[us_train.state == 'California']

    filtered_us = utils.add_taxon_metadata(pth, filtered_us, ARGS.organism)



    # grab only relevant data for training
    print("saving to file")
    filtered_us.to_csv("{pth}/occurrences/occurrences_cali_{org}_census.csv".format(pth=pth, org=ARGS.organism), sep = ';') if ARGS.census else filtered_us.to_csv("{pth}/occurrences/occurrences_cali_{org}_train.csv".format(pth=pth, org=ARGS.organism), sep = ';')

    
if __name__ == "__main__":
    args = ['base_dir', 'organism', 'census']
    ARGS = config.parse_known_args(args)
    main()