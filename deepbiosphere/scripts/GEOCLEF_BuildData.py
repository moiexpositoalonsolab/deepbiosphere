import copy
from geopandas.tools import sjoin
from shapely.geometry import Point, Polygon
import warnings
import geopandas as gpd
import pandas as pd
import numpy as np
import time
import math 
import reverse_geocoder as rg
import glob
import os
from shapely.geometry import Point, Polygon
from deepbiosphere.scripts import GEOCLEF_Utils as utils
from deepbiosphere.scripts import GEOCLEF_Dataset as dataset
from deepbiosphere.scripts import GEOCLEF_Config as config
from deepbiosphere.scripts.GEOCLEF_Config import paths

'''
This file builds the dataset for training the model. The raw data is stored in
"{pth}occurrences/single_obs_{country}_{org}_train.csv"

'''


def get_multiple_joint_from_group(group_df):
    df_np = group_df[['lat', 'lon', 'species', 'gbif_id', 'family', 'genus', 'id']].values #.to_numpy()
    extra_specs = [{df_np[i,2]} for i in range(len(df_np))]
    extra_fams = [{df_np[i,4]} for i in range(len(df_np))] 
    extra_gens = [{df_np[i,5]} for i in range(len(df_np))]
    extra_ids = [{df_np[i,6]} for i in range(len(df_np))]    
    tick = time.time()
    # def nmea_2_meters(lat1, lon1, lat2, lon2):
    for i in range(len(df_np)):
        for j in range(len(df_np)):
            if i != j:
                dist = utils.nmea_2_meters(df_np[i,0], df_np[i,1], df_np[j,0], df_np[j,1])
                if dist <= 256:
                    extra_specs[i].add(df_np[j,2])
                    extra_fams[i].add(df_np[j,4])
                    extra_gens[i].add(df_np[j,5])
                    extra_ids[i].add(df_np[j,6])                    
    tock = time.time()                
    diff = tock - tick
    # ((diff / len(bb_np)) * len(filtered))/(60*60)
    print("took {diff} seconds".format(diff=diff))


    group_df['all_specs'] = extra_specs
    group_df['all_fams'] = extra_fams
    group_df['all_gens'] = extra_gens    
    group_df['extra_ids'] = extra_ids    
    return group_df


def get_single_joint_from_group(group_df):
    
    df_np = group_df[['lat', 'lon', 'species', 'gbif_id', 'family', 'genus', 'id']].values #.to_numpy()
    # create a set with each single observation in it
    extra_specs = [{df_np[i,2]} for i in range(len(df_np))]
    extra_fams = [{df_np[i,4]} for i in range(len(df_np))] 
    extra_gens = [{df_np[i,5]} for i in range(len(df_np))]
    extra_ids = [[-1, set()] for i in range(len(df_np))]
    
    tick = time.time()
    # def nmea_2_meters(lat1, lon1, lat2, lon2):
    for i in range(len(df_np)):
        for j in range(len(df_np)):
            if i != j:
                dist = utils.nmea_2_meters(df_np[i,0], df_np[i,1], df_np[j,0], df_np[j,1])
                if dist <= 256:
                    i_id = df_np[i, 6]
                    j_id = df_np[j,6]
                    # if we haven't seen this cluster of obs before, save one of the obs
                    extra_specs[i].add(df_np[j,2])
                    extra_fams[i].add(df_np[j,4])
                    extra_gens[i].add(df_np[j,5])
                    extra_ids[i][0] = i_id
                    extra_ids[i][1].add(j_id)

    group_df['all_specs'] = extra_specs
    group_df['all_fams'] = extra_fams
    group_df['all_gens'] = extra_gens

    # take out duplicates
    bad_ids = []
    sorted_ids = sorted(extra_ids, reverse=True, key=lambda x: len(x[1]))
    for curr_id, ids in sorted_ids:
        if curr_id in bad_ids:
            continue
        else:
            bad_ids = bad_ids + list(ids)
    group_df = group_df[~group_df.id.isin(bad_ids)]
    tock = time.time()
    diff = tock - tick
    print("took {diff} seconds".format(diff=diff))
    return group_df

def remove_single_location_species(dset):
    # make sure unique location filtering of species
    meme = dset.groupby(['species'])
    tracker = {spec: set() for spec in dset.species.unique()}
    for spec, m in meme:
        print(spec)
        df_np = m[['lat', 'lon', 'species', 'id']].values #.to_numpy()
        for i in range(len(df_np)):
            for j in range(len(df_np)):
                if i != j:
                    dist = utils.nmea_2_meters(df_np[i,0], df_np[i,1], df_np[j,0], df_np[j,1])
                    if dist >= 256:
                        tracker[m.species.iloc[0]].update([df_np[i,3], df_np[j,3]])

    single_ob = [s for s,unq in tracker.items() if len(unq) == 0]     
    cleaned = dset[~dset.species.isin(single_ob)]
    cleaned.index = np.arange(len(cleaned))
    return cleaned


def add_ecoregions(base_dir, dset):
    # convert shapefile to geojson
    # use geopandas to read shapefiles: https://stackoverflow.com/questions/43119040/shapefile-into-geojson-conversion-python-3
    # inspiration: https://github.com/gboeing/urban-data-science/blob/master/19-Spatial-Analysis-and-Cartography/rtree-spatial-indexing.ipynb
    diff = time.time()
    # use geopandas to read shapefiles: https://stackoverflow.com/questions/43119040/shapefile-into-geojson-conversion-python-3
    file = base_dir + 'us_shapefiles/ecoregions3_4/ca/ca_eco_l3.shp'
    shp_file = gpd.read_file(file)
    gdf = gpd.GeoDataFrame(dset)
    gdf['geometry'] = gdf.apply(lambda row: Point((row['lon'], row['lat'])), axis=1)
    gdf.set_crs(epsg=4326, inplace=True)
    gdf = gdf.to_crs(shp_file.crs)
    dset = sjoin(gdf, shp_file, how='left',op="within")    
    doff = time.time()
    print("ecoregions took ", ((doff-diff)/ 60), " minutes")
    return dset
    
    
def create_test_train_split(dset):
    
    # create data structure to track
    # total number of observations of each
    # species in datset, including in other
    # joint observations. Essentially, we know
    # the number of times the species is in an 
    # observation ,but we need to also account for
    # all other observations the species might be
    # appended to.
    tracker = {}
    test = set()
    print("setting up tracker")
    spec_freq = dset.species.value_counts()
    for spec, freq in spec_freq.items():

        # get the other observations this species is a part of the label for
        # to do that, go through all occurrences of the current species
        # and look at how many other species are in the joint label
        # because of the way the joint labels are created, if another species
        # is in the current species joint label, then the current species is 
        # also guaranteed to be in the other species joint label, and thus a 
        # part of that observation. Also, since the current species itself is
        # appended to the joint label, it is sufficient to simply get the number
        # of species in the joint label for all observations of the current species
        # also, since duplicate geospatial locations for an entry are also removed,
        # that guarantees that no edge case where the joint labels across a species' 
        # observations overlap.
        num_obs = 0
        all_obs = dset[dset.species == spec]
        for _, ob in all_obs.iterrows( ):
            num_obs += len(ob.all_specs)
        # the tracker stores the counter, the number of occurences of that species, 
        # and the total # of observations that species is a part of the joint label for
        tracker[spec] = [0,freq,num_obs]
    # keep around a copy of the tracker, will need it later
    trackk = copy.deepcopy(tracker)        
    # now that we have the tracker, let's actually start grabbing indices to add
    # to the test set
    test_idxs = set()
    # this other tracker tracks to make sure that at least one observation for
    # all species have been added to the test set
    loop_tracker = {species: 0 for species in dset.species.unique()}
    freq_spec = spec_freq.keys().to_list()
    num_unq = len(dset.species.unique())
    print("doing first pass through data")
    # essentially, while there is still a species that
    # has no observations yet in the test set, keep adding
    # observations to the test set
    while sum(loop_tracker.values()) < num_unq:
        # randomly grab a species (the distribution of number of 
        # observations per species and the average length of a species'
        # joint label is not uniform, so it's more efficient to randomly sample)
        spec = np.random.choice(freq_spec, 1)[0]
        # if we don't yet have an observation for
        # this species, grab a random observation for
        # that species and add it and all its joint label 
        # observations to the test set
        if loop_tracker[spec] == 0:
            # grab all occurrences of the given species in dataset
            thisspec = dset[dset.species == spec]
            # grab a random occurrence of this species (done for 
            # same reason as above)
            obs = np.random.permutation(thisspec.index.values)
            # grab the ids of all the other occurrences that
            # overlap this occurrence
            extras = dset.iloc[obs[0]].extra_ids
            # now update the species tracker for not just this species but all species from extra_ids
            # To do so, I grab all other occurrences from the joint label
            extra_obs = dset[dset.id.isin(extras)]
            # then, go in and update the tracker for that species
            # the current occurrence's id is included in extra_ids
            for _, ob in extra_obs.iterrows():
                tracker[ob.species][0] += 1
                loop_tracker[ob.species] = 1
            # finally, add all observations to the list
            test_idxs.update(extras)

    # now, every species has at least one observation in the test
    # dataset. However, not every species is represented in the train
    # dataset now, as the above algorithm is a little aggressive and 
    # depending on how a species' joint labels are distributed, all
    # occurrences of that species may have been added to the test
    # dataset. Here, we will re-add all those now-missing species
    # back to the train dataset and then add one observation for
    # each of these species to the test dataset, which will almost
    # perfectly preserve that there is at least one occurrence of
    # each species in train and test
    print("pulling out missing species")
    # using the tracker, see if there are any species for whom all occurrences
    # have been added to test
    certify = [spec for spec in dset.species.unique() if ((tracker[spec][0]) - tracker[spec][1] >= 0)]
    # remove all occurrences associated with those species back into the train set
    to_remove = set()
    for spec in certify:
        rem = dset[dset.species == spec]
        # get all ids for all occurrences associated
        # with this occurrence
        all_occs = [n for m in rem.extra_ids for n in m]
        to_remove.update(all_occs)
    # now move all obs from certify back to train and then re-add species till we get approx min 1 obs
    # per species
    new_test = test_idxs - to_remove
    removed = dset[dset.id.isin(new_test)]
    # figure out what species are now missing from test
    missing = set(dset.species.unique()) - set(removed.species.unique())
    print("refining test set")
    # add to new tracker the already made observations
    for id_ in new_test:
        spec = dset[dset.id == int(id_)]
        trackk[spec.species.values[0]][0] += 1
    # now, go through every missing species and iteratively
    # add one observation for this species to test.
    # We'll be smart though and add the smallest observation to test
    # to make sure the likelihood that we accidentally take all 
    # observations for another species is minimized
    for spec in missing:
        des = dset[dset.species == spec]
        # find smallest obs, add it to test set    
        mins = [(len(d), id_) for d, id_ in zip(des.all_specs, des.id)]
        # find id of occurrence with smallest label
        _, id_ = min(mins, key=lambda x: x[0])
        toadd = des[des.id == id_]
        extras = toadd.extra_ids.tolist()
        # grab all those ids
        extra_obs = dset[dset.id.isin(extras[0])]
        # project if any species would have all observations
        # added to test if we took this observation
        certify = {((trackk[spec][0]+1) - trackk[spec][1] > 0) : spec for spec in extra_obs.species.unique()}
        # if adding this observation would move all occurrences
        # of a given species over to the test set, then don't add
        # this observation
        if sum(certify.keys()) > 0:
            print("species {} only in test set {} with spec {} and length is {}".format(certify[True], trackk[certify[True]], spec, len(new_test)))
            pass
        # if not problematic, then add the observations
        else:
            for _, ob in extra_obs.iterrows():

                trackk[ob.species][0] += 1

            new_test.update(extras[0]) 
    # and now you have a well-balanced test-train split!
    print("train and test set ready!")
    dset['test'] = dset.id.isin(new_test)
#     print(dset.columns)
    print("{} observations in train, {} in test for {}%".format((len(dset)- sum(dset.test)), sum(dset.test), (sum(dset.test)/len(dset)*100) ))
    print("{} species in train, {} species in test".format(len(dset[~dset.test].species.unique()), len(dset[dset.test].species.unique())))
#     return train, test
    return dset
    
def main():
    
    warnings.filterwarnings("ignore")
    print("grab data")
    pth = ARGS.base_dir
    us_train = None
    if ARGS.observation == 'single':
        print("why are you doing this?")
        exit(1)
    us_train_pth = "{}occurrences/single_obs_cali_plant_census.csv".format(pth) if ARGS.census else "{pth}occurrences/single_obs_{country}_{org}_train.csv".format(pth=pth, country=ARGS.region, org=ARGS.organism)
    us_train = pd.read_csv(us_train_pth, sep=None)
    us_train = dataset.reformat_data(us_train)
    if 'species' not in us_train.columns:
        us_train = utils.add_taxon_metadata(pth, us_train, ARGS.organism)
    # remove species with too few observations period
#     print(us_train.columns)»
    spec_freq = us_train.species.value_counts()
#     print(len(spec_freq))
    goodspec = [spec for spec, i in spec_freq.items() if i >= ARGS.threshold]
    us_train = us_train[us_train.species.isin(goodspec)]
    # remove species that are only in one geographic location in dataset
    us_train = remove_single_location_species(us_train)
#     print(len(us_train.species.unique()))
#     exit(1)
    # create a new tuple column
    us_train['lat_lon'] = list(zip(us_train.lat, us_train.lon))
    # convert to list for faster extraction
    us_latlon = us_train['lat_lon'].tolist()
    # grab location data for the lat lon
    res = rg.search(us_latlon)
    # grab necessary info from the results
    states = [r['admin1'] for r in res]
    regions = [r['admin2'] for r in res]
    cities = [r['name'] for r in res]
    # add the states information back into the original dataframe
    us_train['state'] = states
    us_train['region'] = regions
    us_train['city'] = cities
    # group data into smaller, more manageable chunks
    grouped = us_train.groupby(['city', 'state', 'region'])
    write_pth = "{pth}joint_obs/{obs}/".format(pth=pth, obs=ARGS.observation)
    if not os.path.exists(write_pth):
        os.makedirs(write_pth)
    for (grouping, df) in grouped:
        print("grouping {grouping}".format(grouping=grouping))
        if ARGS.observation == 'joint_multiple':
            joint_df = get_multiple_joint_from_group(df)
        elif ARGS.observation == 'joint_single':
            joint_df = get_single_joint_from_group(df)
        else:
            pass


        city = grouping[0].replace(" ", "")
        region = grouping[2].replace(" ", "")
        state = grouping[1].replace(" ", "")
        pth_pth = "{write_pth}{country}_{city}_{region}_{state}.csv".format(write_pth=write_pth, country=ARGS.region, city=city, region=region, state=state)
        joint_df.to_csv(pth_pth)

    # now clean memory and go regrab that data and append
    del us_train, grouped
    glob_pth = "{pth}joint_obs/{obs}/{region}*".format(pth=pth, obs=ARGS.observation, region=ARGS.region)
    all_grouped = glob.glob(glob_pth)
    all_dat = pd.read_csv(all_grouped[0], sep=None)
    for path in all_grouped[1:]:
        new_dat = pd.read_csv(path, sep=None)
        all_dat = pd.concat([all_dat, new_dat])
    # make sure data is the right format
    all_dat = dataset.reformat_data(all_dat)
    # and save data 
    print("moving to ecoregions")
    # add ecoregions data
    all_dat = add_ecoregions(pth, all_dat)
    all_dat = create_test_train_split(all_dat)
    train_pth = "{pth}/occurrences/{obs}_obs_{region}_{plant}_{train}_{threshold}.csv".format(obs=ARGS.observation, pth=pth, region=ARGS.region, plant=ARGS.organism,train='train', threshold=ARGS.threshold)
#     test_pth = "{pth}/occurrences/{obs}_obs_{region}_{plant}_{test}_{threshold}.csv".format(obs=ARGS.observation, pth=pth, region=ARGS.region, plant=ARGS.organism,test='test', threshold=ARGS.threshold
    all_dat.to_csv(train_pth)
#     all_test.to_csv(test_pth)
    
    
if __name__ == "__main__":
    
    args = ['base_dir', 'organism', 'census', 'region', 'observation', 'threshold']
    ARGS = config.parse_known_args(args)
    main()
