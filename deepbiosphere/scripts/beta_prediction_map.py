import torch
import multiprocessing as mp
from tqdm import tqdm
# deepbio packages
from deepbiosphere.scripts.GEOCLEF_Config import paths
from multiprocessing import Process
import deepbiosphere.scripts.NAIP_Utils as naip
import deepbiosphere.scripts.GEOCLEF_Utils as utils



if __name__ == '__main__':

    mp.set_start_method('spawn')
    # setting up variables and datasets
    shpfile = naip.Load_NAIP_Bounds(paths.NAIP_BASE, 'ca', '2012')
    state = 'ca'
    year = '2012'
    ca_tifb =f"{state}/{year}/{state}_100cm_{year}/"
    modelname = 'old_tresnet_satonly'
    nodata = -9999
    div = 'beta'
    warp = False
    base_dir = paths.AZURE_DIR
# grab the relevant rasters
    rasters = [f"{base_dir}inference/prediction/raw/{modelname}/{ca_tifb}{fman.APFONAME[:5]}/{'_'.join(fman.FileName.split('_')[:-1])}.tif" for _, fman in shpfile.iterrows()]


# TODO: split rasters into K pieces and assign to each process
    K = 24 # 24 cpus on cluster, best to use
    par = utils.partition(rasters, K)
# split off predictions to cpus and launch subprocesses
# ctx = mp.get_context('spawn')
    procs = []
    for rasters, _ in zip(par, range(K)):
        procs.append(Process(target=naip.diversity_raster_list, args=(rasters, div, year, base_dir, modelname, warp, nodata)))


    for i, p in enumerate(procs):
        # maybe an ordering thing: https://stackoverflow.com/questions/46755640/spawning-multiple-processes-with-python
        print(f"starting dev {i} process")
        p.start()
        print(f"process for dev {i} started")
