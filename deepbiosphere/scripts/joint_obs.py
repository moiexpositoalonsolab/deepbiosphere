import warnings
import pandas as pd
import time
import math 
import reverse_geocoder as rg
import glob
import os
from deepbiosphere.scripts import GEOCLEF_Utils as utils
from deepbiosphere.scripts import GEOCLEF_Config as config
from deepbiosphere.scripts.GEOCLEF_Config import paths




def get_multiple_joint_from_group(group_df):
    df_np = group_df[['lat', 'lon', 'species', 'gbif_id', 'family', 'genus']].values #.to_numpy()
    extra_specs = [{df_np[i,2]} for i in range(len(df_np))]
    extra_fams = [{df_np[i,4]} for i in range(len(df_np))] 
    extra_gens = [{df_np[i,5]} for i in range(len(df_np))]
    
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
    tock = time.time()                
    diff = tock - tick
    # ((diff / len(bb_np)) * len(filtered))/(60*60)
    print("took {diff} seconds".format(diff=diff))


    group_df['all_specs'] = extra_specs
    group_df['all_fams'] = extra_fams
    group_df['all_gens'] = extra_gens    
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




def main():
    
    warnings.filterwarnings("ignore")
    print("grab data")
    pth = ARGS.base_dir
    us_train = None
    if ARGS.observation == 'single':
        print("why are you doing this?")
        exit(1)
    us_train_pth = "{}occurrences/single_obs_cali_plant_census.csv".format(pth) if ARGS.census else "{pth}occurrences/single_obs_{country}_{org}_train.csv".format(pth=pth, country=ARGS.region, org=ARGS.organism)
    us_train = pd.read_csv(us_train_pth, sep=';')
    us_train = utils.add_taxon_metadata(pth, us_train, ARGS.observation)
    
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
#     import pdb; pdb.set_trace()
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
    all_dat = pd.read_csv(all_grouped[0])
    for path in all_grouped[1:]:
        new_dat = pd.read_csv(path)
        all_dat = pd.concat([all_dat, new_dat])
    # and save data 
    pth = "{pth}/occurrences/{obs}_obs_{region}_{plant}_train_census.csv".format(obs=ARGS.observation, pth=pth, region=ARGS.region, plant=ARGS.organism) if ARGS.census else "{pth}/occurrences/{obs}_obs_{region}_{plant}_{train}.csv".format(obs=ARGS.observation, pth=pth, region=ARGS.region, plant=ARGS.organism,train='train')
    all_dat.to_csv(pth)
    
    
if __name__ == "__main__":
    
    args = ['base_dir', 'organism', 'census', 'region', 'observation']
    ARGS = config.parse_known_args(args)
    main()
