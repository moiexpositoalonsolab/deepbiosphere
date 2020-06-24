from random import randrange
import pandas as pd
import argparse
import numpy as np
import random
import math
from deepbiosphere.scripts import GEOCELF_CNN as cnn
from deepbiosphere.scripts import GEOCELF_Dataset as Dataset
from deepbiosphere.scripts import paths
import reverse_geocoder as rg
print("getting data")
pth = paths.DBS_DIR
us_train_pth = "{}occurrences/occurrences_us_train.csv".format(pth)
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
filtered_us = us_train[us_train.state == 'California']



## getting family, genus, species ids for each observation
# get all relevant files
print("adding taxon information")
gbif_meta = pd.read_csv("{}occurrences/species_metadata.csv".format(pth), sep=";")
taxons = pd.read_csv("{}occurrences/Taxon.tsv".format(pth), sep="\t")
# get all unique species ids in filtered train data
us_celf_spec = filtered_us.species_id.unique()
# get all the gbif species ids for all the species in the us sample
conversion = gbif_meta[gbif_meta['species_id'].isin(us_celf_spec)]
gbif_specs = conversion.GBIF_species_id.unique()
# get dict that maps CELF id to GBIF id
spec_2_gbif = dict(zip(conversion.species_id, conversion.GBIF_species_id))
filtered_us['gbif_id'] = filtered_us['species_id'].map(spec_2_gbif)
# grab all the phylogeny mappings from the gbif taxons file for all the given species
# GBIF id == taxonID
taxa = taxons[taxons['taxonID'].isin(gbif_specs)]
phylogeny = taxa[['taxonID', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus']]
gbif_2_king = dict(zip(phylogeny.taxonID, phylogeny.kingdom))
gbif_2_phy = dict(zip(phylogeny.taxonID, phylogeny.phylum))
gbif_2_class = dict(zip(phylogeny.taxonID, phylogeny['class']))
gbif_2_ord = dict(zip(phylogeny.taxonID, phylogeny.order))
gbif_2_fam = dict(zip(phylogeny.taxonID, phylogeny.family))
gbif_2_gen = dict(zip(phylogeny.taxonID, phylogeny.genus))
filtered_us['family'] = filtered_us['gbif_id'].map(gbif_2_fam)
filtered_us['genus'] = filtered_us['gbif_id'].map(gbif_2_gen)
filtered_us['order'] = filtered_us['gbif_id'].map(gbif_2_ord)
filtered_us['class'] = filtered_us['gbif_id'].map(gbif_2_class)
filtered_us['phylum'] = filtered_us['gbif_id'].map(gbif_2_phy)
filtered_us['kingdom'] = filtered_us['gbif_id'].map(gbif_2_king)
cali_plant = filtered_us[filtered_us.kingdom == 'Plantae']

# grab only relevant data for training
print("saving to file")
cali_plant.to_csv("{}/occurrences/occurrences_cali_plants.csv".format(pth), sep = ';')
