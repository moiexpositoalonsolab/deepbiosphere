# deepbiosphere functions
from deepbiosphere.Utils import paths
from deepbiosphere import Utils as utils

# misc functions
import os
import re
import json
import time
import zipfile
import requests
import argparse
import datetime

# countries is a list of countries, states is a list of GADM GIDs which are constructed as
# country code from https://en.wikipedia.org/wiki/ISO_3166-1_alpha-3  [code].[state number]_1
# where state number is the alphabetical sorting of states
# TODO: use GADM to resolve state / country names to their administrative area ids
def request_gbif_records(gbif_user, gbif_email, organism, start_date="2015", end_date="2022", area=['USA.5_1'], wkt_geometry=None):

    # confirm email roughly matches email shape
    if not re.match(r"[^@]+@[^@]+\.[^@]+", gbif_email):
        raise ValueError("It looks like {} is not a valid email address!".format(gbif_email))


    # create download predicate json file
    down_pred = {
        'creator' : gbif_user,
        "notificationAddresses": [
            gbif_email
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
            # only grab observations with <= 120 m uncertainty
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
                "value": "120.0",
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
    if organism == 'animal':

        taxon_json = {
                    "type": "equals",
                    "key": "TAXON_KEY",
                    "value": "1",
                    "matchCase": False
                }
    elif organism == 'plant':
        taxon_json = {
                    "type": "equals",
                    "key": "TAXON_KEY",
                    "value": "6", 
                    "matchCase": False
                }
    elif organism == 'bacteria':
        taxon_json = {
                    "type": "equals",
                    "key": "TAXON_KEY",
                    "value": "3",
                    "matchCase": False
                }
    elif organism == 'plantanimal':
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
    if not wkt_geometry: # For now, only use GADM GIDs if we don't directly pass a WKT geometry
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
    else:
        area_json = {
            "type": "within",
            "geometry": wkt_geometry
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
    # use a pre-made occurrence directory for placing observations


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
        req = requests.get("https://api.gbif.org/v1/occurrence/download/{}".format(id.text))
        curr_time = datetime.datetime.now()
        savepath = f"{paths.OCCS}{organism}_{start_date}_{end_date}_{area[0].replace('.','_')}_acq{curr_time.year}_{curr_time.month}_{curr_time.day}"
        savelink = f"{savepath}.zip"
        download_url(req.json()['downloadLink'], savelink)
        print("data successfully saved to {}".format(savelink))
    elif resp.json()['status'] == "CANCELLED":
        raise ValueError("data download was cancelled :(")
    else:
        raise ValueError("unkown server response encountered: {}".format(resp.json()['status']))
    # unzip files from gbif
    i = 0
    with zipfile.ZipFile(savelink, 'r') as zip_ref:
        # names = the unique DOI GBIF assigns to each download
        names = zip_ref.namelist()
        currdir = f"{os.path.dirname(savepath)}/"
        if len(names) == 1:
            # and use savepath
            zip_ref.extractall(currdir)
            os.rename(currdir + names[0], f"{savepath}.csv")
        else:
            for n in names:
                zip_ref.extractall(path=dirr, members=[n])
                os.rename(currdir + n, f"{savepath}_{i}.csv")
                i += 1
        
    name = f"{savepath}.json"
    down_pred['response'] = resp.json()
    with open(name, 'w') as f:
        json.dump(down_pred, f)    


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

if __name__ == "__main__":

    args = argparse.ArgumentParser()
    args.add_argument('--gbif_user', type=str, required=False, default=os.getenv("GBIF_USER"), help='Gbif user id')
    args.add_argument('--gbif_email', type=str, required=False, default=os.getenv("GBIF_EMAIL"), help='Email address associated with gbif account')
    args.add_argument('--organism', type=str, required=True, help='What organism/s to download', choices=['bacteria', 'plant','animal','plantanimal'])
    args.add_argument('--start_date', type=str, help='Collect observations on and after this year', default='2015')
    args.add_argument('--end_date', type=str, help='Collect observations on and before this year', default='2022')
    args.add_argument('--area', type=str, help='GADM area code for where observations should be taken from', default=['USA.5_1'])
    args.add_argument('--wkt_geometry', type=str, required=False, help='WKT geometry for where observations should be taken from', default=None)
    args, _ = args.parse_known_args()

    request_gbif_records(**vars(args))