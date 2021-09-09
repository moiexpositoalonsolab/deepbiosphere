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
    # vals
    naip_dir = paths.NAIP_BASE
    base_dir = paths.AZURE_DIR
    modelname = 'old_tresnet_satonly'
    state=  'ca'
    warp = False
    year='2012'
    res = 256
    model_pth='nets/tresnet_m_lr0.0001_e11.tar'
    means = means =  (111.27668932654558, 115.84299163319858, 104.88420063186129, 132.9687599226994) # TODO: fix
    cfg_pth='joint_multiple_plant_cali_TResNet_M_AsymmetricLossOptimized_satellite_only_tresnet_m.json'
    shpfile = naip.Load_NAIP_Bounds(naip_dir, state, year)
    fnames = []
    ca_tifb =f"{state}/{year}/{state}_100cm_{year}/" # TODO: fix resolution stupid thing
    tif_dir = naip_dir + ca_tifb
    print(tif_dir)
    for _, fman in shpfile.iterrows():
        fnames.append(naip.Grab_TIFF(fman, tif_dir))
# find open GPUs with enough memory
    avail_gpus = torch.cuda.device_count()
    par = utils.partition(fnames, avail_gpus)
    print(f"{avail_gpus} GPUs are available")
# split off tiffs to gpus and launch subprocesses
# ctx = mp.get_context('spawn')
    procs = []
    for p, d in zip(par, range(avail_gpus)):
        procs.append(Process(target=naip.predict_raster_list, args=(d, p, modelname, res, year,means, model_pth, cfg_pth, base_dir, warp)))


    for i, p in enumerate(procs):
        # maybe an ordering thing: https://stackoverflow.com/questions/46755640/spawning-multiple-processes-with-python
        print(f"starting dev {i} process")
        p.start()
        print(f"process for dev {i} started")
