#!/bin/bash

# all of this stuff is already in the deepbiophere repo so no need to do hopefully, but uncomment if you want to build fully from scratch
#mkdir GeoCLEF
#cd GeoCLEF
#mkdir occurrences
#cd occurrences
# 1. readme / metadata for datasets
#wget -O readme.txt "https://aicrowd-production.s3.eu-central-1.amazonaws.com/task_dataset_files/clef_task_23/fa5427e5-d7a6-4d3b-9f29-6a6fbe61d2a5_README.txt?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJ6IZH6GWKDCCDFAQ%2F20200429%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20200429T230452Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=9fe6020b2abad9ba64b7e90950bbd1eb19f35f13568e5300a828e708fba8d420"
#2. species occurrences 
#wget -O occurences.zip "https://aicrowd-production.s3.eu-central-1.amazonaws.com/task_dataset_files/clef_task_23/99296675-be09-44aa-bbfc-c25b8dbf2ee7_Occurrences.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJ6IZH6GWKDCCDFAQ%2F20200429%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20200429T230452Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=00a8ea7c5b89f38ae2ed15299da08f97f2dd78fb076c3321d13d267fea39a17c"
# 3. environmental rasters
#wget -O rasters.zip "https://aicrowd-production.s3.eu-central-1.amazonaws.com/task_dataset_files/clef_task_23/397e7091-2422-4c6e-92cf-940fa7da52c8_rasters.zip?X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAJ6IZH6GWKDCCDFAQ%2F20200429%2Feu-central-1%2Fs3%2Faws4_request&X-Amz-Date=20200429T230452Z&X-Amz-Expires=604800&X-Amz-SignedHeaders=host&X-Amz-Signature=c1a24d295d9f25f6d0c407173a974932b57d61e0357543c5a7e3bc98c3248a9b"

#unzip occurences.zip
#unzip rasters.zip

# make sure you have parallel downloaded and installed!
# follow these instructions to get parallel

cd GeoCELF2020/

cd occurrences/
wget https://hosted-datasets.gbif.org/datasets/backbone/2019-09-06/backbone.zip
unzip backbone.zip
cd ..
mkdir patches_us
cd patches_us
parallel --eta -v -a us_files.txt wget
# wget -i us_files.txt
parallel tar xvfz ::: *.gz
#parallel unzip ::: *.tar.gz*
# for file in *.tar.gz*; do tar -zxvf "$file"; done
cd ..
mkdir patches_fr
cd patches_fr
parallel --eta -v -a french_files.txt wget
# wget -i french_files.txt
parallel unzip ::: '*.zip*'
# unzip '*.zip*'
echo "data successfully downloaded"
