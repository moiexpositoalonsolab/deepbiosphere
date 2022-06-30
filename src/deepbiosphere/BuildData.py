from shapely.geometry import Point, Polygon, MultiPolygon, LineString, box
import multiprocessing
import argparse
from tqdm import tqdm
from deepbiosphere.Utils import paths
import deepbiosphere.NAIP_Utils  as naip
import deepbiosphere.GEOCLEF_Utils as utils
import deepbiosphere.Dataset as dataset
import json
import rasterio
from rasterio.windows import Window
from scipy.spatial import cKDTree
from collections import Counter
import geopandas as gpd
import pandas as pd
import torch
import glob
import numpy as np
import math
import time
import os
'''
Code for generating the base csv used in all subsequent analyses.
This file adds overlapping species, adds spatial splits, determines
what points are valid from the underlying rasters and image data,
and adds all that information to the csv for future reference
'''


# conversion factor for latitude (longitude can change)
# https://stackoverflow.com/questions/5217348/how-do-i-convert-kilometres-to-degrees-in-geodjango-geos
DEG_2_KM = 0.008 # degrees to kilometers, 1 km aprox for latitude only

def make_test_split(daset, res, latname, loname, excl_dist, rng, idCol='gbifID', frac=.1):
    # rasterio pixels are up to 1200 m across
    # so be sure to grab points with no data leakage
    assert (res>=0.09) and (res <=30), "resolution should be in meters!"
    overlap_dist = math.ceil(256*res)

    daset = daset.to_crs(naip.M_CRS_1)
    tock = time.time()
    # will again use the ckdtree trick to
    # streamline nearest neighbor search
    nA = np.array(list(daset.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(daset.geometry.apply(lambda x: (x.x, x.y))))
    print("building cKDTree")
    btree = cKDTree(nB)
    # max number of unique observations expected in a 256m radius
    K = 2000
    # sqrt(2) for max size of bioclim pixels
    print("querying cKDTree")
    dist, idx = btree.query(nA, k=K)
    tick = time.time()
    print(f"took {(tick-tock)/60} minutes to load KDtree for all {len(daset)} points")
    # first get all the overlapping ids
    overlapping_ids = daset.overlapping_ids.tolist()
    # then map the unique gbif IDs to their index in the array
    gbif_2_ind = { k:v for k, v in zip(daset[idCol], np.arange(len(daset)))}
    # and reverse map it
    ind_2_gbif = { v:k for k, v in gbif_2_ind.items()}
    # for tracking how many observations go together into a cluster
    train_clusters, test_clusters = {}, {}
    train, test = [], []
    nc_train, nc_test = 0, 0
    # keep track of observations we've already assigned
    seen = np.array([False]*len(daset))
    next_dists = np.array([-1.0]*len(daset))
    cluster_name = np.array([-1]*len(daset))
    for i in tqdm(range(len(daset)), total=len(daset), unit=' observation'):
        # if we've already added the cluster,
        # ignore it
        if seen[i]:
            continue
        # include all idxs of immediately overlapping observations
        # since we included the ids of the current observation
        # in the overlapping set, shouldn't need to add the current
        # obs' id as well
        curr = set([gbif_2_ind[a] for a in overlapping_ids[i]])
        prev = curr
        all_id = set()
        # keep adding observations
        # so long as there are outstanding
        # overlapping neighbors that haven't
        # been added
        while len(prev) != 0:
            # get the set of all overlapping neighbors for the unexplored items
            # then unroll it and turn it into a set
            curr = set([gbif_2_ind[val] for sublist in [overlapping_ids[k]for k in prev] for val in sublist])
            # find what id's haven't been checked yet
            prev = curr-all_id
            # and add those ids to the list
            all_id.update(prev)
        # now that we have the ids grouped, we need to figure out what is the distance
        # to the next-nearest non-overlapping observation that's not in the cluster
        # for each clustered observation
        subd = dist[list(all_id),:]
        subi = idx[list(all_id),:]
        dists = []
        # go through the index of each observation's
        # closest neighbors
        for k, row in enumerate(subi):
            # look and see what indices aren't present.
            # The next-nearest id not present in the
            # cluster is the nearest neighbor
            # so get the distance to that neighbor
            for j,id_ in enumerate(row):
                if id_ not in all_id:
                    dists.append(subd[k,j])
                    break
                # so we're guaranteed from earlier
                # that all 256 or less obs will be in the
                # joint observation, but when clustering
                # observations together, some observations
                # may be so daset that you can't capture all of them
                # with the nearest 2K neighbors...
                # for exmple, some clusters have 17K obs in
                # the cluster. That should definitely be training set
                # and if any obs is  >256 but <1300 that wasn't adde
                # it'll get aded to the train set later, so good to ignore
                elif j == 1999:
                    dists.append(subd[k,j])
        # find the shortest distance to the nearest neighbor
        next_dist = min(dists)
        next_dists[list(all_id)] = dists
        # and save them for future use
        next_dists[list(all_id)] = dists
        if next_dist >excl_dist:
            nc_test +=1
            test += list(all_id)
            cluster_name[list(all_id)] = nc_test
            # save the size, min distance to, and approximate location of this cluster
            test_clusters[nc_test] = [next_dist, len(all_id), daset.iloc[i][latname], daset.iloc[i][loname]]
        else:
            nc_train +=1
            cluster_name[list(all_id)] = nc_train
            # save the size, min distance to, and approximate location of this cluster
            train_clusters[nc_train] = [next_dist, len(all_id), daset.iloc[i][latname], daset.iloc[i][loname]]
            train += list(all_id)
        seen[list(all_id)] = True
    daset['unif_train_test'] = None
    daset['unif_train_test'].iloc[test] = 'test'
    daset['unif_train_test'].iloc[train] = 'train'
    daset['cluster_dist'] = next_dists
    # finally, get distance to neighboring observationt
    daset['neighbor_dist'] = np.take_along_axis(dist,np.expand_dims((dist<=overlap_dist).sum(axis=1), axis=1), axis=1)
    # finally, find distance to next-nearest
    daset['cluster_assgn'] = cluster_name

    # if a large portion of the dataset is
    # spatially removed, only keep a random subset
    if len(daset[daset.unif_train_test == 'test']) > (len(daset)*frac):
        print(f"trimming down test set from {round((len(daset[daset.unif_train_test == 'test'])/len(daset)*100), 3)}% of dataset to {frac*100}%")
        # randomly remove some clusters from test
          # and keep frac% of them
        while len(daset[daset.unif_train_test == 'test']) > (len(daset)*frac):
            # remove test clusters and add to train one at a time
            to_remove = rng.permutation(np.array(list(test_clusters.keys())))[0]
            daset.loc[(daset.unif_train_test == 'test') & (daset.cluster_assgn == to_remove), 'unif_train_test']  = 'train'
            train_clusters[to_remove] = test_clusters[to_remove]
            del test_clusters[to_remove]

    return daset, train_clusters, test_clusters

def save_data(daset, year, means, tr_clus, te_clus, sp, gen, fam, daset_id, count_spec, count_gen, count_fam, idCol, latname, loname):
    print("saving data!")
    filepath = f"{paths.OCCS}{daset_id}"
    # theoretically we should remove useless columns
    # as well, but will save that to be manual for now
    # get overlapped ids
    # might also grab extra leftover cols from the
    # original dataset that should be cleaned out
    ids = [c for c in daset.columns if '_id' in c]
    # get names of overlapping taxa
    names = [c for c in daset.columns if 'overlapping' in c]
    epa_regions = ['US_L3NAME', 'NA_L3NAME', 'NA_L2NAME', 'NA_L1NAME']
    files = [c for c in daset.columns if 'file' in c]
    # banded test train split
    splits = [f"train_{i}" for i in range(10)] +[f"test_{i}" for i in range(10)]
    tokeep = [idCol, latname, loname, 'unif_train_test',  'cluster_dist', 'len_overlap', 'cluster_assgn', 'species', 'family', 'genus', 'order', 'neighbor_dist', 'APFONAME', 'UTM'] + ids + names + epa_regions  + files + splits
    daset[tokeep].to_csv(f"{filepath}.csv")
    # json can't serialize numpy dtypes, annoying...
    sp = {k:v.item() for k,v in sp.items()}
    gen = {k:v.item() for k,v in gen.items()}
    # if this dataset has already been generated once
    # and we're just adding filenames for a new year
    fam = {k:v.item() for k,v in fam.items()}
    all_dat = {
        'dataset_means' : means,
        'train_clusters' : tr_clus,
        'test_clusters' : te_clus,
        'spec_2_id' : sp,
        'gen_2_id' : gen,
        'fam_2_id' : fam,
        'species_counts' : count_spec,
        'genus_counts' : count_gen,
        'family_counts' : count_fam
    }

    with open(f"{filepath}_metadata.json", 'w') as f:
        json.dump(all_dat, f, indent=4)

def calculate_means_parallel(rasters, procid, lock, year):
    means = []
    stds = []

    with lock:
        prog = tqdm(total=len(rasters), desc=f"calculating means with proc {procid}", unit=' tiffs', position=procid)
    for i, ras in enumerate(rasters):
        src = rasterio.open(ras)
        dat = src.read()
        dat = torch.tensor(dat)
        # first scale raster from 0-255 to 0-1
        dat = utils.scale(dat, out_range=(0, 1), min_=0, max_=255)
        # then calculate mean and std dev and save
        means.append(torch.mean(dat,dim=[1,2]).tolist())
        stds.append(torch.std(dat,dim=[1,2]).tolist())
        with lock:
            prog.update(1)
    with lock:
        prog.close()
    return means, stds

def calculate_means(tiff_dset_name, parallel, year, rasters=None):
   # Decision: going to do it acros all satellite image, not all images in the dataset
    # load in previously generated means
    f = f"{paths.OCCS}dataset_means.json"
    with open(f, 'r') as fp:
        daset_means =  json.load(fp)
    if rasters is None:
        rasters = glob.glob(f"{paths.SCRATCH}{tiff_dset_name}/*/m*.tif")
    # now, chunk up the rasters into K sections
    ras_pars = utils.partition(rasters, parallel)
    # TQDM for parallel processes: https://stackoverflow.com/questions/66208601/tqdm-and-multiprocessing-python
# parallel process the rasters
    lock = multiprocessing.Manager().Lock()
    pool =  multiprocessing.Pool(parallel)
    res_async = [pool.apply_async(calculate_means_parallel, args=(ras, i, lock, year)) for i, ras in enumerate(ras_pars)]
    res_dfs = [r.get() for r in res_async]
    pool.close()
    pool.join()
# result is a list of tuples for each chunk of rasters
# so, separate the  tuples into list of list of means
    means, stds = zip(*res_dfs)
# and flatten list of tuples
    means = [val for sublist in means for val in sublist]
    stds = [val for sublist in stds for val in sublist]
    mean = torch.mean(torch.stack(torch.tensor(means)), dim=0)
    std = torch.mean(torch.stack(torch.tensor(stds)), dim=0)
    print(len(means), len(stds), mean.shape, std.shape)
    mean = mean.tolist()
    std = std.tolist()
    daset_means[f"naip_{year}"]['means'] = mean
    daset_means[f"naip_{year}"]['stds'] = std
    return daset_means

def map_key(df, key, new_key=None):
    key_2_id = {
        k:v for k, v in
        zip(df[key].unique(), np.arange(len(df[key].unique())))
    }
    if new_key == None:
        df[key] = df[key].map(key_2_id)
    else:
        df[new_key] = df[key].map(key_2_id)
    return df, key_2_id

# return dataframe with
# 1. individual species mapped to 1-N value
# 2. overlapping observation mapped to one-hot
def map_to_index(daset):

    # map species name to 0-N id variable
    daset, spec_2_id = map_key(daset, 'species', 'species_id')
    daset, gen_2_id = map_key(daset, 'genus', 'genus_id')
    daset, fam_2_id = map_key(daset, 'family', 'family_id')

    # and save indices
    daset['specs_overlap_id'] = [[spec_2_id[a] for a in a_s] for a_s in daset.overlapping_specs]
    daset['gens_overlap_id'] = [[gen_2_id[a] for a in a_s] for a_s in daset.overlapping_gens]
    daset['fams_overlap_id'] = [[fam_2_id[a] for a in a_s] for a_s in daset.overlapping_fams]

    return daset, spec_2_id, gen_2_id, fam_2_id

def add_filenames(daset, state, year, tiff_dset_name, idCol='gbifID'):
    # get shapefile for that year (should only be 1 so can use glob to resolve)
    shp_pth = glob.glob(f"{paths.SHPFILE}naip_tiffs/{state}_shpfl_{year}/*shp")[0]
    tif_shps = gpd.read_file(shp_pth)
    # also set up what is the current directory based on year and state
    # tiny bit hacky but the shapefiles from the gov don't have the resolution
    # saved in a consistent format across years so for now this is easiest
    # first, make sure in same crs
    daset = daset.to_crs(tif_shps.crs)
    # now, fix the FileName column since most
    # years only keep the acquisition date
    # and ignore the second date which corresponds
    # to the date the image was re-released
    if str(year) != '2018':
        tif_shps[f'corr_filename_{year}'] = [f"{f.rsplit('_', 1)[0]}.tif" for f in tif_shps.FileName]
    # *except* that in 2018 they used the entire
    # FileName, so there's no need to modify it
    else:
        tif_shps[f'corr_filename_{year}'] = tif_shps.FileName
    # since each NAIP tiff is designed to have a "bleed" zone of 128-300 pixels that
    # it shares with other neighboring tiffs to ensure no seams or gaps between images,
    # we can be ensured that if an observation point lies within a tif outline in tif_shps
    # even if it's right on the edge, there's enough pixels left in the bleed zone to acquire
    # an image for the observation. Therefore, we can just use the spatial join between the
    # observations and the tif boundaries to determine which observation/s go with which tiffs.
    # Note: default operation is "intersects" which considers two geometires as overlappping if
    # their boundary or interior intersect. Because we have the bleed zone, even if an observation
    # is on the boundary of the tif outline, there's still plenty of pixels to actually extract
    # the whole image, which is why we use "intersects" not "within"
    combined = gpd.sjoin(daset, tif_shps)
    # if an observation is on the edge between multiple tifs, then sjoin will make a new row
    # for each tif that observation intersects with. Just throw away the extra rows corresponding
    # to these duplicate observations (again, doesn't matter which tif we pick because the actual
    # tif at least 150-300 pixels wider and taller than the shpfile suggests,
    # so just greedily take  the first row)
    combined.drop_duplicates(subset=idCol, inplace=True)
    # also we don't care about the index from the tifs dataframe, so dump that too
    del combined['index_right']
    # and finally add the numpy image filepath
    print('||| columns: ',[c for c in combined.columns if 'file' in c])
    combined[f"filepath_{year}"] = [f"{tiff_dset_name}/{apfo[:5]}/{(corr_filename).split('.')[0]}.npz" for corr_filename, apfo in zip(combined[f'corr_filename_{year}'], combined.APFONAME)]
    return combined

# extract and save an image for each observation in the dataset
def make_images_parallel(daset:gpd.GeoDataFrame, year, tiff_dset_name, procid, lock, idCol='gbifID'):
    # now that we've got the filename for each observation, we can simply
    # group observations by tiff, read off each image for each observation
    # from the tiff and save to a npz archive for easy access later on!
#     daset[f"filepath_{year}"] = None
    daset[f'imageproblem_{year}'] = False
    with lock:
        prog = tqdm(total=len(daset), desc=f"adding images to proc {procid}", unit=' observations', position=procid)
    for fname, df in daset.groupby(f'corr_filename_{year}'):
        images = {}
        # file structure is {state}_{resolution}cm_{year}/APFONAME (first 5 digits)/filename
        apfo = df.APFONAME.unique()
        assert len(apfo) == 1, "multiple APFOs per-image!"
        fpath = f"{tiff_dset_name}/{apfo[0][:5]}/{fname}"
        fullpath = f"{paths.SCRATCH}{fpath}"
#         fullpath = f"{paths.SCRATCH}{df[f'filepath_{year}'].iloc[0]}"
        src = rasterio.open(fullpath)
        for i, obs in df.iterrows():
            # have to use iloc on the original df to make sure that
            # the observation is aligned to the correct crs
            # we do the entire dataset at a time because conversion
            # is slow and it's faster to amortize over many tiffs than
            # per-tif
            if int(str(src.crs).split(':')[-1]) != df.crs.to_epsg():
                daset = daset.to_crs(src.crs)
                x, y = daset.loc[i].geometry.xy
            else:
                x, y = daset.loc[i].geometry.xy
            # get the row/col starting location of the point in the raster
            xx,yy = rasterio.transform.rowcol(src.transform, x,  y)
            # rasterio returns arrays, collapse down to ints
            xx,yy = xx[0],yy[0]
            # read data from a 256x256 window centered on observation
            image_crop = src.read(window=Window(yy-128, xx-128, 256, 256))
            if (image_crop.shape[1] == 256) and (image_crop.shape[2] == 256):
                # save image to dict where key is idCol (unique for all obs)
                images[f"{obs[idCol]}"] = image_crop
            else:
                daset[f'imageproblem_{year}'][i] = True
            with lock:
                prog.update(1)
        # finally, store image archive
        savename = f"{fpath.split('.')[0]}.npz"
#         daset[f"filepath_{year}"][df.index] = savename
        savepath = f"{paths.IMAGES}{savename}"
        # if there are already some images stored,
        # only add new images that weren't there already in the keydict
        if os.path.exists(savepath):
            # open file & check what keys are in archive against above
            curr = np.load(savepath)
            missing = [k for k in images.keys() if k not in curr.keys()]
            # then resave everything if there is any new image to add
            if len(missing) > 0:
                curr = dict(np.load(savepath).items())
                for k in missing:
                    curr[k] = images[k]
                np.savez(savepath, **curr)
        else:
            # if this is the first time building the archive
            # make the directories and save out to disk
            currdir = os.path.dirname(savepath)
            if not os.path.exists(currdir):
                os.makedirs(currdir)
            np.savez(savepath, **images)
    with lock:
        prog.close()
    # drop any obs where the image wasn't saved for any reason
    daset = daset[~daset[f'imageproblem_{year}']]
    return daset
# extract and save an image for each observation in the dataset
def make_images(daset:gpd.GeoDataFrame, year, tiff_dset_name, idCol='gbifID'):
    # now that we've got the filename for each observation, we can simply
    # group observations by tiff, read off each image for each observation
    # from the tiff and save to a npz archive for easy access later on!
    daset[f"filepath_{year}"] = None
    daset[f'imageproblem_{year}'] = False
    prog = tqdm(total=len(daset), desc="adding images", unit=' observations')
    for fname, df in daset.groupby(f'corr_filename_{year}'):
        images = {}
        # file structure is {state}_{resolution}cm_{year}/APFONAME (first 5 digits)/filename
        apfo = df.APFONAME.unique()
        assert len(apfo) == 1, "multiple APFOs per-image!"
        fpath = f"{tiff_dset_name}/{apfo[0][:5]}/{fname}"
        fullpath = f"{paths.SCRATCH}{fpath}"
        src = rasterio.open(fullpath)
        for i, obs in df.iterrows():
            # have to use iloc on the original df to make sure that
            # the observation is aligned to the correct crs
            # we do the entire dataset at a time because conversion
            # is slow and it's faster to amortize over many tiffs than
            # per-tif
            if int(str(src.crs).split(':')[-1]) != df.crs.to_epsg():
                daset = daset.to_crs(src.crs)
                x, y = daset.loc[i].geometry.xy
            else:
                x, y = daset.loc[i].geometry.xy
            # get the row/col starting location of the point in the raster
            xx,yy = rasterio.transform.rowcol(src.transform, x,  y)
            # rasterio returns arrays, collapse down to ints
            xx,yy = xx[0],yy[0]
            # read data from a 256x256 window centered on observation
            image_crop = src.read(window=Window(yy-128, xx-128, 256, 256))
            if (image_crop.shape[1] == 256) and (image_crop.shape[2] == 256):
                # save image to dict where key is idCol (unique for all obs)
                images[f"{obs[idCol]}"] = image_crop
            else:
                daset[f'imageproblem_{year}'][i] = True
            prog.update(1)
        # finally, store image archive
        savename = f"{fpath.split('.')[0]}.npz"
        daset[f"filepath_{year}"][df.index] = savename
        savepath = f"{paths.IMAGES}{savename}"
        # if there are already some images stored,
        # only add new images that weren't there already in the keydict
        if os.path.exists(savepath):
            # open file & check what keys are in archive against above
            curr = np.load(savepath)
            missing = [k for k in images.keys() if k not in curr.keys()]
            # then resave everything if there is any new image to add
            if len(missing) > 0:
                curr = dict(np.load(savepath).items())
                for k in missing:
                    curr[k] = images[k]
                np.savez(savepath, **curr)
        else:
            # if this is the first time building the archive
            # make the directories and save out to disk
            currdir = os.path.dirname(savepath)
            if not os.path.exists(currdir):
                os.makedirs(currdir)
            np.savez(savepath, **images)
    prog.close()
    # drop any obs where the image wasn't saved for any reason
    daset = daset[~daset[f'imageproblem_{year}']]
    return daset

# for each observation, get all the other
# observations in a K m radius and append
# to observation. Also, filter out species
# with not enough observations (below threhsold)
def add_overlapping_filter(daset, res, threshold=200, idCol='gbifID'):

    # first, depending on the resolution, calculate
    # the "nearby" radius (256 for 1m, ~154 for 60cm)
    # maxar data can be up to 9cm resolution and
    # landsat is 30m so use that as a reasonable range to confirm
    # resolution is in meters
    assert (res>=0.09) and (res <=30), "resolution should be in meters!"
    overlap_dist = math.ceil(256*res)
    # convert to a 1m resolution crs
    # covers half the state but can live
    # with the slight distortion
    daset = daset.to_crs(naip.M_CRS_1)
    tock = time.time()
    # will again use the ckdtree trick to
    # streamline nearest neighbor search
    nA = np.array(list(daset.geometry.apply(lambda x: (x.x, x.y))))
    nB = np.array(list(daset.geometry.apply(lambda x: (x.x, x.y))))
    print("building cKDTree")
    btree = cKDTree(nB)
    # max number of unique observations expected in a 256m radius
    K = 2000
    print("querying cKDTree")
    dist, idx = btree.query(nA, k=K)
    assert (dist <= overlap_dist).sum(axis=1).max() < K, f"more than {K} observations overlapping in dataset! Please increase K"
    tick = time.time()
    print(f"took {(tick-tock)/60} minutes to load KDtree for all {len(daset)} points")
    # overlap_spec is used to keep track of how many times a
    # a species has been observed jointly, and overlap_id
    # is used for mapping remaining observations
    overlap_spec, overlap_id = [], []
    specids = np.array(daset[idCol].tolist())
    specs = np.array(daset.species.tolist())
    # grab the nearby unique overlapping species for each observation
    # here, we define "overlapping" as any species observed
    # within a 256m radius. Technically, this bleeds over
    # the edge of a given image for some observations (256x256 pixels)
    # but ecological theory and experimental evidence supports that
    # interactions and co-occurrence is still strongly driven by
    # individuals within a few hundred meters, so 256 is valid
    for i, (row, close) in tqdm(enumerate(zip(dist, idx)),total=len(daset), desc="adding overlapping observations", unit=' observations'):
        # select the rows from the dataset corresponding to
        # the observations within a 256m radius of the current point
        # this will also grab the current observation as well
        overlap_spec.append(specs[close[row<=overlap_dist]])
        overlap_id.append(specids[close[row<=overlap_dist]])

    # finally, filter out species below the threshold
    # by first counting the number of observations
    # per-species in the joint observations
    flattened = [val for sublist in overlap_spec for val in sublist]
    # get the counts of each species across dataset
    counts = Counter(flattened)
    # and get the names of species that should be
    # removed because they're below the threshold
    to_remove = [s for s, c in counts.items() if c < threshold]

    # filter down to only the observations associated with the
    # species with enough observations to remain
    # the count of co-occurring species in these observations
    # will also change, and the minimum number of obs for a species
    # may be slightly below the threshold as a function of the removal
    # process, but it shouldn't be too terribly many observations removed
    # so remaining species should still be reasonably close to the threshold
    remaining = daset[~daset.species.isin(to_remove)]
    # now, map observation id to species only for species above the threshold
    id_2_spec = {i:s for i,s in zip(remaining[idCol],remaining.species)}
    # and also map to genus and family
    id_2_gen = {i:g for i,g in zip(remaining[idCol], remaining.genus)}
    id_2_fam = {i:f for i,f in zip(remaining[idCol], remaining.family)}
    # now, filter out ids of observations that have been removed (not present in above dict)
    overlap_id = [list(filter(lambda id_: id_ in id_2_spec.keys(), curr)) for curr in overlap_id]
    # then map observation id to species
    # and also ids to genus; family
    overlap_spec = [[id_2_spec[id_] for id_ in curr] for curr in overlap_id]
    overlap_gen = [[id_2_gen[id_] for id_ in curr] for curr in overlap_id]
    overlap_fam = [[id_2_fam[id_] for id_ in curr] for curr in overlap_id]
    # now, save these results to dataframe
    daset['overlapping_specs'] = overlap_spec
    daset['overlapping_gens'] = overlap_gen
    daset['overlapping_fams'] = overlap_fam
    daset['overlapping_ids'] = overlap_id
    daset['len_overlap'] = [len(o) for o in overlap_spec]
    # and finally filter out observations with species below threshold again
    daset = daset[~daset.species.isin(to_remove)]
    # and get spec, gen, fam id counts
    flat_spec = [val for sublist in daset.overlapping_specs for val in sublist]
    flat_gen = [val for sublist in daset.overlapping_gens for val in sublist]
    flat_fam = [val for sublist in daset.overlapping_fams for val in sublist]
    # get the final counts of species, genus, and family
    count_spec = Counter(flat_spec)
    count_gen = Counter(flat_gen)
    count_fam = Counter(flat_fam)
    return daset, count_spec, count_gen, count_fam


def add_ecoregions(dframe, idCol):
    diff = time.time()
    file = f"{paths.SHPFILE}ecoregions/ca/ca_eco_l3.shp"
    shp_file = gpd.read_file(file)
    shp_file = shp_file.to_crs(dframe.crs)
    # default join is "intersects"
    # "An object is said to intersect other if its boundary
    # and interior intersects in any way with those of the other."
    # since we don't care if a point is on the exact boundary
    # versus interior of a ecoregion, will just go with the
    # intersects default
    print("adding ecoregions")
    daset = gpd.sjoin(dframe, shp_file)
    # some points lie outside of the ecoregions dataframe
    # (usually on the border of the pacific ocean)
    # still keep those points, but the ecoregion will be nan
    missing = set(dframe[idCol])- set(daset[idCol])
    missing = dframe[dframe[idCol].isin(missing)]
    daset = pd.concat([daset,missing])
    daset = gpd.GeoDataFrame(daset, geometry=daset.geometry, crs=naip.NAIP_CRS)
    # also we don't care what the indx
    # of the ecoregion was now that
    # we have it, so can get rid of it
    del daset['index_right']
    doff = time.time()
    print(f"ecoregions took {(doff-diff)/ 60} minutes to add ecoregions")
    return daset



def filter_raster_oob(daset,rasters):
    # go through each location and query the raster for that point
    # TODO: this code assumes it's a bioclim only set of rasters
    daset['out_of_bounds'] = False
    daset = daset.to_crs(naip.NAIP_CRS) # bioclim rasters are guaranteed to be WSG84 crs
    # then, go through and append the bioclim variables to the csv
    for i, row in tqdm(daset.iterrows(),total=len(daset),desc='checking points inside raster', unit='  observations'):
        for j, (ras, transf) in enumerate(rasters):
        # go through each location and query the raster for that point
            x,y = rasterio.transform.rowcol(transf, *row.geometry.xy)
            if np.ma.is_masked(ras[0,x,y]):
                daset['out_of_bounds'][i] = True
                # no need going through other rasters bc out of bounds
                break
    # filter out those out-of-bounds points
    daset = daset[~daset.out_of_bounds]
    return daset

# Basically copying code from below that returns back just the polygons for visualization
def generate_split_polygons():
    # these are a box around california
    # leaves a bit of a buffer around
    # the whole state
    lonmin, lonmax=-125,-114
    lowlat, highlat=32,42.1
    # add buffer region around min, max latitude that's guaranteed to capture all
    # points in the state
    # TODO: make this a parameter so it generalizes
    strtlat, endlat = 32, 42
    exclude_size = DEG_2_KM+DEG_2_KM*0.5 # max largest size of bioclim pixel is sqrt(2) ~1.5 km
    polys = {}
    for i, lat in enumerate(range(strtlat, endlat, 1)):

        # polygon for below exclusion band
        train_top  = [Point(lonmax, lat+1), Point(lonmax, highlat), Point(lonmin, highlat),  Point(lonmin, lat+1)]
        train_top = Polygon(train_top)
        # polygon for above the exclusion band
        train_bot = [Point(lonmax, lowlat), Point(lonmax, lat), Point(lonmin, lat),  Point(lonmin, lowlat)]
        train_bot = Polygon(train_bot)
        # polygon for test locations
        # exclude_size is the buffer
        test = [Point(lonmax, lat+exclude_size), Point(lonmax, lat+1-exclude_size), Point(lonmin, lat+1-exclude_size),  Point(lonmin, lat+exclude_size)]
        test = Polygon(test)
        exclude_bot = [Point(lonmax, lat), Point(lonmax, lat+exclude_size), Point(lonmin, lat+exclude_size),  Point(lonmin, lat)]
        exclude_bot =  Polygon(exclude_bot)
        exclude_top = [Point(lonmax, lat+1-exclude_size), Point(lonmax, lat+1), Point(lonmin, lat+1),  Point(lonmin, lat+1-exclude_size)]
        exclude_top = Polygon(exclude_top)

        polys[f"band_{i}"] = {
                    'train' : [train_top, train_bot],
                    'test' : test,
                    'exclusion' : [exclude_top, exclude_bot],
                }
    return polys

# make bands and exclusion zones
def make_spatial_split(daset, latCol):
    # first, make sure we're in the right crs
    daset = daset.to_crs(naip.NAIP_CRS)
    # these are a box around california
    # leaves a bit of a buffer around
    # the whole state
    lonmin, lonmax=-125,-114
    lowlat, highlat=32,42.1
    # iterate through the lat/lons in the dataset
    # iterate through this so we don't add extra bands
    # from buffer radius above
    # want to start either at the 32 degree mark
    # or whatever latitude the most southern obs is
    strtlat = max(lowlat, math.floor(daset[latCol].min()))
    # want to end at either the 42 degree mark
    # or if all points are more than a degree lower, that
    endlat = min(math.floor(highlat), math.ceil(daset[latCol].max()))
    for i, lat in enumerate(range(strtlat, endlat, 1)):

        exclude_size = DEG_2_KM+DEG_2_KM*0.5 # max largest size of bioclim pixel is sqrt(2) ~1.5 km
        # polygon for below exclusion band
        train_top  = [Point(lonmax, lat+1), Point(lonmax, highlat), Point(lonmin, highlat),  Point(lonmin, lat+1)]
        train_top = Polygon(train_top)
        # polygon for above the exclusion band
        train_bot = [Point(lonmax, lowlat), Point(lonmax, lat), Point(lonmin, lat),  Point(lonmin, lowlat)]
        train_bot = Polygon(train_bot)
        # polygon for test locations
        # exclude_size is the buffer
        test = [Point(lonmax, lat+exclude_size), Point(lonmax, lat+1-exclude_size), Point(lonmin, lat+1-exclude_size),  Point(lonmin, lat+exclude_size)]
        test = Polygon(test)
        # get all the points inside train bands
        train_1 = daset[daset.intersects(train_bot)]
        train_2 = daset[daset.intersects(train_top)]
        train_pts = pd.concat([train_1, train_2]) if i != 0 else train_2
        train_pts = gpd.GeoDataFrame(train_pts, geometry=train_pts.geometry, crs=train_1.crs)
        # save which points are in train split for this band
        daset[f"train_{i}"] = False
        daset[f"train_{i}"][train_1.index] = True
        daset[f"train_{i}"][train_2.index] = True
        # get all points in test bands
        test_pts = daset[daset.intersects(test)]
        # save which points are in test split for this band
        daset[f"test_{i}"] = False
        daset[f"test_{i}"][test_pts.index] = True
        # just sanity check there's no overlap
        # https://gis.stackexchange.com/questions/222315/finding-nearest-point-in-other-geodataframe-using-geopandas
        nA = np.array(list(test_pts.geometry.apply(lambda x: (x.x, x.y))))
        nB = np.array(list(train_pts.geometry.apply(lambda x: (x.x, x.y))))
        btree = cKDTree(nB)
        dist, idx = btree.query(nA, k=1)
        print(f"{len(train_pts)} training points, {len(test_pts)} testing points,  {round(len(train_pts)/len(daset)*100, 3)}% train, {round(len(test_pts)/len(daset)*100,3)}% test, {round(min(dist)/DEG_2_KM, 3)} kilometers between test and train")
    return daset

    # here, if a species is singleton (all observation of that
    # species can be found in the same 256m radius) then
    # remove that species from the dataset (can't learn a good
    # representation with only one observation). Furthermore,
    # remove any duplicate observations (observations within 150m of each other)
    # of the same species
def remove_singletons_duplicates(daset, res):
    # first, convert geometry to
    # UTM11 so distances are in m
    daset = daset.to_crs(naip.M_CRS_1)
    # next, get the distance for the resolution we'll be working with
    assert (res>=0.09) and (res <=30), "resolution should be in meters!"
    overlapped = math.ceil(256*res)
    # also, remove any species with fewer than, like 4 observations
    # cause it's not gonna stay anyway and screws up later code
    vc = daset.species.value_counts()
    daset = daset[daset.species.isin(vc[vc> 4].keys())]
    to_remove = []
    to_drop = []
    for name, group in tqdm(daset.groupby('species'),total=len(daset.species.unique()),desc='removing singletons and duplicate observations', unit=' species'):
        # use R-Tree trick and compute distance
        # between locations to each other, pairwise
        nA = np.array(list(group.geometry.apply(lambda x: (x.x, x.y))))
        nB = np.array(list(group.geometry.apply(lambda x: (x.x, x.y))))
        btree = cKDTree(nB)
        dist, idx = btree.query(nA, k=len(nA))
        # if there's a row where every other obs is within 256m
        # which means the check of <256m will eval to true for each
        # element in the row, so we can just check the sum of that
        # bool to be the length of the number of observations
        if (dist[:,1:] <= overlapped).sum(axis=1).max() == len(group):
            to_remove.append(name)
        # look at each observation
        # and remove any other observation
        # within 150m. (150 because it'll approximately
        # keep the pixels associated with each duplicate
        # observation still in the dataset)
        dups = []
        ignore = []
        for i in range(len(group)):
            # if this observation is removed
            # as a duplicate or has been visited before
            # we're ensured
            # that this location has already been
            # checked for duplicates
            if (i in dups) or (i in ignore):
                continue
            curr_dist = dist[i]
            n_dups = (curr_dist<=math.ceil(150*res)).sum()
            curr_idx = idx[i,:n_dups]
            # if there is another observation besides
            # the current observation that is within
            # 75m of itself, remove the other close
            # observation
            if n_dups == 2 and (curr_dist[1]<math.ceil(75*res)):
                dups+= curr_idx[1:].tolist()
                ignore += [curr_idx[0]]
            # otherwise, if there are more than 2 duplicate
            # observations in a location, keep the furthest
            # and the closest observation to reflect some of
            # the density in the dataset
            elif n_dups > 2:
                dups+= curr_idx[1:n_dups-1].tolist()
                # only keep the closest and farthest obs
                ignore += [curr_idx[0], curr_idx[-1]]
        to_drop += group.index[dups].tolist()
    # remove duplicate observation rows
    daset = daset.drop(index=to_drop)
    # remove singleton species
    daset = daset[~daset.species.isin(to_remove)]
    return daset


# generate the csv
def make_dataset(dset_path, daset_id, latname, loname, sep, year, state, threshold, rng, idCol, parallel, add_images, only_images, excl_dist, outline=f"{paths.SHPFILE}gadm36_USA/gadm36_USA_1.shp", to_keep=None):
    daset = pd.read_csv(dset_path, sep=sep)
    pts = [Point(lon, lat) for lon, lat in zip(daset[loname], daset[latname])]
    # GBIF returns coordinates in WGS84 according to the API
    # https://www.gbif.org/article/5i3CQEZ6DuWiycgMaaakCo/gbif-infrastructure-data-processing
    daset = gpd.GeoDataFrame(daset, geometry=pts, crs=naip.NAIP_CRS)
    # also make the name of the directory where the images are stored
    # it's a weird structure and so we'll do it a hacky way for now
    tiff_dset_name = f"{state}_100cm_{year}" if str(year) in ['2012', '2014'] else f"{state}_060cm_{year}"
    # filter down to just vascular plants
    if 'class' not in daset.columns:
#         us_train = utils.add_taxon_metadata(pth, us_train, ARGS.organism)
        raise NotImplementedError
    #  gbif does keep subspecies but not varieties and we will as well
    # tried using the wcvp but they're not aligned
    # some species spelled different, etc
    # https://www.gbif.org/dataset/f382f0ce-323a-4091-bb9f-add557f3a9a2
    vasculars = [
        'Gnetopsida',
        'Liliopsida',
        'Lycopodiopsida',
        'Magnoliopsida',
        'Pinopsida',
        'Polypodiopsida',
        'Lycopodiopsida',
        'Ginkgoopsida'
    ]
    daset = daset[daset['class'].isin(vasculars)]
    # if there's a list of species to keep provided
    # go ahead and filter down now
    if to_keep is not None:
          daset = daset[daset.species.isin(to_keep)]
    # and read in state outline
    us1 = gpd.read_file(outline) # the state's shapefiles
    ca = us1[us1.NAME_1 == 'California']
    # ensure dataframes are in the same crs
    ca = ca.to_crs(naip.NAIP_CRS)
    # keep only points inside of GADM california
    if 'index_right' in daset.columns:
        del daset['index_right']
    daset = gpd.sjoin(daset, ca, op='within')
    # remove leftover index from ca shapefile
    del daset['index_right']
    # get resolution depending on what year it is
    # before 2015 was 1 meter, after was 60 cm
    res = 1.0 if year in ['2012', '2014'] else 0.6
    # this boolean allows us to just add the images if we so desire
    # keep only the points inside of rasters
    rasters = dataset.get_bioclim_rasters()
    daset = filter_raster_oob(daset, rasters)
    # add ecoregion
    daset = add_ecoregions(daset, idCol)
    # remove species only in one location in dataset
    # also remove extra duplicate observations
    daset = remove_singletons_duplicates(daset, res)
    # next, add the filename of the tiff that corresponds with each obs
    daset = add_filenames(daset, state, year, tiff_dset_name, idCol)
    if add_images and (parallel == 0):
        # then, make images and keep only points inside of NAIP imagery
        # do so serially (all images generated by one thread)
        daset = make_images(daset, year, tiff_dset_name, idCol)
    elif add_images and (parallel > 0):
        # first, add the filename of the tiff that corresponds with each obs
        # then, sort by filename to group observations together
        daset.sort_values(by=[f'corr_filename_{year}'], inplace=True)
        # now, chunk up the dataset into K sections
        idx_pars = utils.partition(range(len(daset)), parallel)
        procs = []
        # TQDM for parallel processes: https://stackoverflow.com/questions/66208601/tqdm-and-multiprocessing-python
        lock = multiprocessing.Manager().Lock()
        pool =  multiprocessing.Pool(parallel)
        res_async = [pool.apply_async(make_images_parallel, args=(daset.iloc[idxs], year, tiff_dset_name, i, lock, idCol)) for i, idxs in enumerate(idx_pars)]
        res_dfs = [r.get() for r in res_async]
        pool.close()
        pool.join()
# finally, moosh all the dataframes back together
# theoretically, the indices should be respected...
        daset = pd.concat(res_dfs)
        daset = gpd.GeoDataFrame(daset, geometry=daset.geometry)
    #if not only_images:
    # add joint observations
    # and remove species without enough obs
    daset, count_spec, count_gen, count_fam = add_overlapping_filter(daset, res, threshold, idCol)

    # add 10 spatial bands
    daset = make_spatial_split(daset, latname)
    # add local exclusions.
    # with the spatial autocorrelation in the dataset,
    # we really only usually have enough points for one
    # set of test points to keep data leakage out
    daset, train_clusters, test_clusters = make_test_split(daset, res,  latname, loname, excl_dist, rng,  idCol)
    # map species, genus, family to universal index
    daset, sp,gen,fam = map_to_index(daset)
    # get the means for this naip dataset
#     means = calculate_means(tiff_dset_name, parallel, args.year)
    means = None
    # and finally save everything out to disk
    save_data(daset, year, means, train_clusters, test_clusters, sp, gen,fam, daset_id, count_spec, count_gen, count_fam, idCol, latname, loname)

if __name__ == "__main__":
    # set up argparser
    args = argparse.ArgumentParser()


    args.add_argument('--dset_path', type=str, required=True, help='Absolute path to base dataset to use')
    args.add_argument('--species_file', type=str, help='If you want to filter down the observations to a pre-defined set of species, give the filename to a json with the specie name as a list here', default=None)
    args.add_argument('--daset_id', type=str, required=True, help='What to call the newly generated dataset')
    args.add_argument('--latname', type=str, help='Name of the column that contains latitude information', default='decimalLatitude')
    args.add_argument('--loname', type=str, help='Name of the column that contains latitude information', default='decimalLongitude')
    args.add_argument('--sep', type=str, required=True, help='The separator used to delimeter the base dataset')
    args.add_argument('--year', type=str, help='What year of NAIP to use to build the dataset')
    args.add_argument('--state', type=str, help='What state to build the dataset in', default='ca')
    args.add_argument('--excl_dist', type=int, help='How far away to exclude data points in the uniform split', default=1300)
    args.add_argument('--threshold', type=int, help='Minimum number of observations required to keep a species in the dataset', default=200)
    args.add_argument('--add_images', dest='add_images', action='store_true', help='Set this if you want to generate images for the observations')
    args.add_argument('--only_images', dest='only_images', action='store_true', help='Set this if you *only* want to generate images, ie: you have already run the other cleaning parts of the script')
    args.add_argument('--parallel', type=int, default=0, help='Number of parallel processes to use if parallelizing image download')
    args.add_argument('--idCol', type=str, required=True, help="What column to use as the unique identfier for observations")
    args.add_argument('--seed', type=int, default=0)
    args, _ = args.parse_known_args()
    # if seed has been set, use
    # else just use system defaults
    if args.seed >= 0:
        rng =  np.random.default_rng(args.seed)
    else:
        rng = np.random.default_rng()
    if args.parallel > 0:
        multiprocessing.set_start_method("spawn")
    if args.species_file is not None:
        with open(args.species_file, 'r') as f:
            species = json.load(f)
        make_dataset(args.dset_path, args.daset_id, args.latname, args.loname, args.sep, args.year, args.state, args.threshold, rng, args.idCol, args.parallel, args.add_images, args.only_images, args.excl_dist, to_keep=species)
    else:
        make_dataset(args.dset_path, args.daset_id, args.latname, args.loname, args.sep, args.year, args.state, args.threshold, rng, args.idCol, args.parallel, args.add_images, args.only_images, args.excl_dist)
