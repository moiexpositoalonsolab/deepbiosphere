# todo:
# 1. build a script that downloads all pointwise plant data for a given timeframe from gbif?
# so we can only download large chunks of data asynchronously
# so what I'll do is provide the download predicate
# and a tutorial notebook that walks through a user of how to download
# their own gbif occurrence records
# but for myself,
    # a. make a download predicate
    # b. instantiate an asynchronous gbif download request for all the current california data
    # c. retreive the requested data
    # d. move it into place and update the repository framework to support it
    # e. automate this whole process for potential end users
# 2. build a script to download all the tiffs for the known lat / lon from the new gbif data
    # a. will include building a new tree directory to store the images in
    # b. will also probably need to re-mount a new storage drive
    # c. will also need to link gbif ids to image ids
# 3. build a script to download all the tiffs for all the locations in california oh boy
    # or alternately, see if can asynch do it with cloud-optimized geotiffs and rasterio?

    # oh no wait... build a python script that builds the correct json predicate and prints the curl to call
    # that way it should be very easy to swap out states, etc!

import os
import re
import json
import time
import zipfile
import requests
from deepbiosphere.scripts import GEOCLEF_Utils as utils
from deepbiosphere.scripts import GEOCLEF_Config as config
from deepbiosphere.scripts.GEOCLEF_Config import paths
# countries is a list of countries, states is a list of GADM GIDs which are constructed as
# country code from https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3  [code].[state number]_1
# where state number is the alphabetical sorting of states
def request_gbif_records(base_dir, gbif_usr, email, taxon, start_date="2015", end_date="2021", area=['USA.5_1']):

    # confirm email roughly matches email shape
    if not re.match(r"[^@]+@[^@]+\.[^@]+", email):
        raise ValueError("It looks like {} is not a valid email address!".format(email))


    # TODO: convert taxon to json code
    # create download predicate json file
    down_pred = {
        'creator' : gbif_usr,
        "notificationAddresses": [
            email
        ],
        "sendNotification": True,
        "format": "SIMPLE_CSV",
        "predicate": {
            "type": "and", 
            "predicates": [
            # only grab human-annotated observations
            {
                "type": "equals",
                "key": "BASIS_OF_RECORD",
                "value": "HUMAN_OBSERVATION",
                "matchCase": False
            },
            # only grab observations with <= 30 m uncertainty
            {
            "type": "and",
            "predicates": [
                {
                "type": "greaterThanOrEquals",
                "key": "COORDINATE_UNCERTAINTY_IN_METERS",
                "value": "0.0",
                "matchCase": False
                },
                {
                "type": "lessThanOrEquals",
                "key": "COORDINATE_UNCERTAINTY_IN_METERS",
                "value": "30.0",
                "matchCase": False
                }
            ]
            },
            # only get observations that have lat / lon coordinates
            {
                "type": "equals",
                "key": "HAS_COORDINATE",
                "value": "True",
                "matchCase": False
            },
            # only get observations that don't have geospatial issues  
            {
                "type": "equals",
                "key": "HAS_GEOSPATIAL_ISSUE",
                "value": "False",
                "matchCase": False
            },
            # don't care about absence datapoints
            {
                "type": "equals",
                "key": "OCCURRENCE_STATUS",
                "value": "present",
                "matchCase": False
            },
            ]
        }
    }

    taxon_json = None
    # get the taxon and add it to json
    if taxon == 'animal':

        taxon_json = {
                    "type": "equals",
                    "key": "TAXON_KEY",
                    "value": "1",
                    "matchCase": False
                }
    elif taxon == 'plant':
        taxon_json = {
                    "type": "equals",
                    "key": "TAXON_KEY",
                    "value": "6", 
                    "matchCase": False
                }
    elif taxon == 'bacteria':
        taxon_json = {
                    "type": "equals",
                    "key": "TAXON_KEY",
                    "value": "3",
                    "matchCase": False
                }
    elif taxon == 'plantanimal':
        # do a group join
        taxon_json = {
            "type": "or",
            "predicates": [
                {
                "type": "equals",
                "key": "TAXON_KEY",
                "value": "6",
                "matchCase": False
                },
                {
                "type": "equals",
                "key": "TAXON_KEY",
                "value": "1",
                "matchCase": False
                }
            ]
    }

    down_pred['predicate']['predicates'].append(taxon_json)


    # get the correct state / country
    # going to use gadm gids
    # will maybe also include the option for country
    area_json = {}
    if len(area) == 1:
        area_json = {
            "type": "equals",
            "key": "GADM_GID",
            "value": area[0],
            "matchCase": False
        }
    else:
        area_json = {
            "type" : "or",
            "predicates":[{
                "type": "equals",
            "key": "GADM_GID",
            "value": a,
            "matchCase": False
            } for a in area]
        }
    down_pred['predicate']['predicates'].append(area_json)
    


    # get the correct time range
    dates = {
        "type": "and",
        "predicates": [
        {
            "type": "greaterThanOrEquals",
            "key": "YEAR",
            "value": start_date,
            "matchCase": False
        },
        {
            "type": "lessThanOrEquals",
            "key": "YEAR",
            "value": end_date,
            "matchCase": False
        }
        ]
    }
    down_pred['predicate']['predicates'].append(dates)
    # now create an occurrences folder from helper function in files script
    config.setup_main_dirs(base_dir)
    dirr = base_dir + 'occurrences/'

    # curl --include --user userName:PASSWORD --header "Content-Type: application/json" --data @query.json https://api.gbif.org/v1/occurrence/download/request
    # need: pass user / pass
    # users should have their GBIF credentials stored in a .netrc file, specifially at ~/.netrc
    # the structure of the netrc should be
    # machine api.gbif.org
    # login <your gbif username>
    # password <your pass>
    # example here https://stackoverflow.com/questions/6031214/git-how-to-use-netrc-file-on-windows-to-save-user-and-password

    header = {'Content-Type' : "application/json"}
    # https://api.gbif.org/v1/occurrence/download/
    id = requests.post("https://api.gbif.org/v1/occurrence/download/request", json=down_pred)
    # import pdb; pdb.set_trace()
    print("download id is {}".format(id.text))
    if not id.ok:
        raise ValueError(id.text)

        # print("Request failed. Download size likely too large")
    resp = requests.get("https://api.gbif.org/v1/occurrence/download/{}".format(id.text))
    print("response status ", resp.json()['status'])
    while resp.json()['status'] == 'RUNNING' or resp.json()['status'] == 'PREPARING':
        print("sleeping for 5 minutes before checking download again")
        time.sleep(60*5) #*5 sleep thread for 5 minutes before re-querying server
        resp = requests.get("https://api.gbif.org/v1/occurrence/download/{}".format(id.text))
    
    if resp.json()['status'] == "SUCCEEDED":
        meme = requests.get("https://api.gbif.org/v1/occurrence/download/{}".format(id.text))
        savelink = dirr + config.build_gbif_file(taxon, start_date, end_date, area, 'zip')
        download_url(meme.json()['downloadLink'], savelink)
        print("data successfully saved to {}".format(savelink))
    elif resp.json()['status'] == "CANCELLED":
        raise ValueError("data download was cancelled :(")
    else:
        raise ValueError("unkown server response encountered: {}".format(resp.json()['status']))
    # unzip files from gbif
    i = 0
    with zipfile.ZipFile(savelink, 'r') as zip_ref:
        names = zip_ref.namelist()
        print( names)
        if len(names) == 1:
            zip_ref.extractall(dirr)
            os.rename(dirr + names[0], dirr + config.build_gbif_file(taxon, start_date, end_date, area, 'csv'))
        else:
            for n in names:
                zip_ref.extractall(path=dirr, members=[n])
                os.rename(dirr + n, dirr + config.build_gbif_file(taxon, start_date, end_date, area, 'csv') + str(i))
                i += 1
        # import pdb;pdb.set_trace()
        
        name = dirr + config.build_gbif_file(taxon, start_date, end_date, area, 'json')
    down_pred['response'] = resp.json()
    with open(name, 'w') as f:
        json.dump(down_pred, f)    
    # so the request returns an id, if you curl from https://api.gbif.org/v1/occurrence/download/<id> then it returns a json with the status of the download
    # eventually, it'll return success and the downloadLink field of that json will be populated
    # TODO: figure out what the id gives you
    # then query the server every 5 minutes until the data is ready
    # then download it to the occurrences folder

def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

if __name__ == "__main__":
    request_gbif_records(paths.AZURE_DIR, 'gillespl', 'gillespl@cs.stanford.edu', 'plant', start_date="2015", end_date="2021", area=['USA.5_1'])