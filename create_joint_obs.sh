#!/bin/bash



# first, create joint_obs directory
# need to be in GeoCLEF2020/ directory


CURR_PARAMS=""
while [[ $# -gt 0 ]]
do
key="$1"

case $key in
-base_dir)
BASE_DIR="$2"
# CURR_PARAMS=" $CURR_PARAMS $key $2 " # add to list of params to include in Aagos run
shift # shift past curr argument
shift # shift past curr value
;; # indicates end of case

    *)    # unknown option
    echo "ERROR: unknown command line argument: "$1" not a recognized command"
    exit 1
    ;;
esac
done

pwd
cd GeoCLEF/
mkdir joint_obs

echo $BASE_DIR

python3 ../deepbiosphere/scripts/joint_obs.py --country us --base_dir $BASE_DIR
python3 ../deepbiosphere/scripts/joint_obs.py --country fr --base_dir $BASE_DIR
cd joint_obs/
head -1 us_Abbeville_HenryCounty_Alabama.csv > joint_obs_us.csv ; tail -n +2 -q us*.csv >> joint_obs_us.csv
head -1 fr_Abbeville_DepartementdelaSomme_Picardie.csv > joint_obs_fr.csv ; tail -n +2 -q fr*.csv >> joint_obs_fr.csv
cp joint_obs_us.csv ../occurrences/
cp joint_obs_fr.csv ../occurrences/
