import numpy as np
import math
import pandas as pd
import glob
import torch
import matplotlib.pyplot as plt
# adding this to check github integration on slack
# TODO: move general methods into here

def cnn_output_size(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    return(output)

def plot_image(base_dir, id_, figsize=(10,10), transpose=False):
    imgs = image_from_id(id_, base_dir)
    fig, axs = plt.subplots(3, figsize=figsize) if transpose else plt.subplots(1, 3, figsize=figsize)
    axs[0].imshow(np.transpose(imgs[1:4,:,:], [1,2,0]))
    axs[0].set_title("rgb ")
    axs[1].imshow(imgs[0,:,:].squeeze())
    axs[1].set_title("altitude")
    axs[2].imshow(imgs[4:5,:,:].squeeze())
    axs[2].set_title("infrared")
    return fig
    
def num_corr_matches(output, target):
    tot_acc = []
    acc_acc = []
    for obs, trg in zip(output, target):
        out_vals, out_idxs = torch.topk(obs, int(trg.sum().item()))
        targ_vals, targ_idxs = torch.topk(trg, int(trg.sum().item()))
        eq = len(list(set(out_idxs.tolist()) & set(targ_idxs.tolist())))
        acc = eq / trg.sum() * 100
        tot_acc.append((eq, len(targ_idxs)))
        acc_acc.append(acc.item())
    
    return np.stack(acc_acc), np.stack(tot_acc)

# https://gist.github.com/weiaicunzai/2a5ae6eac6712c70bde0630f3e76b77b        
def topk_acc(output, target, topk=(1,), device=None):
    """Computes the standard topk accuracy for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    targ = target.unsqueeze(1).repeat(1,maxk).to(device)
    correct = pred.eq(targ)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).cpu()
        res.append(correct_k.mul_(100.0 / batch_size))
    del targ, pred, target
    return res

def image_from_id(id_, base_dir):
    # make sure image and path are for same region
    cdd, ab, cd = id_2_file(id_)
    subpath = "patches_{}/{}/{}/".format('fr', cd, ab) if id_ >= 10000000 else "patches_{}/patches_{}_{}/{}/{}/".format('us', 'us', cdd, cd, ab)
    return subpath_2_img(base_dir, subpath, id_)

def subpath_2_img(pth, subpath, id_):
    alt = "{}{}{}_alti.npy".format(pth, subpath, id_)
    rgbd = "{}{}{}.npy".format(pth, subpath, id_)    
    # Necessary because some data corrupted...
    try:
        np_al = np.load(alt)
        np_img = np.load(rgbd)
    except KeyboardInterrupt:
        print("operation cancelled")
        exit(1)
    except:
        print("trouble loading file {}, faking data :(".format(rgbd))
        # magic numbers 173 and 10000000 are first files in both us and fr datasets
        channels, height, width = get_shapes(173, pth) if id_ < 10000000 else get_shapes(10000000, pth)
        np_al = np.zeros([height, width], dtype='uint8') 
        np_img = np.zeros([channels-1, height, width], dtype='uint8')
        np_img = np.transpose(np_img, (1,2,0))
    np_al = np.expand_dims(np_al, 2)
    np_all = np.concatenate((np_al, np_img), axis=2)
    return np.transpose(np_all,(2, 0, 1))

def id_2_subdir(id_):
    return id_2_subdir_fr(id_) if id_ >=  10000000 else id_2_subdir_us(id_)

    
def id_2_file(id_):
    return id_2_file_fr(id_) if id_ >= 10000000 else id_2_file_us(id_)
    
def id_2_file_us(id_):
    abcd = id_ % 10000
    ab, cd = math.floor(abcd/100), abcd%100
    cdd = math.ceil((cd+ 1)/5)
    cdd = "0{}".format(cdd)  if cdd < 10 else "{}".format(cdd)
    ab = "0{}".format(ab) if id_ / 1000 > 1 and ab < 10 else ab
    cd = "0{}".format(cd) if  cd < 10 else cd
    return cdd, ab, cd

def id_2_file_fr(id_):
    abcd = id_ % 10000
    ab, cd = math.floor(abcd/100), abcd%100
    ab = "0{}".format(ab) if id_ / 1000 > 1 and ab < 10 else ab
    cd = "0{}".format(cd) if  cd < 10 else cd
    return None, ab, cd
    

'''files are default assumed to be ';' separator '''
def check_gbif_files(occ_paths, img_path, sep=';'):
    occs = []
    for path in occ_paths:
        occs.append(pd.read_csv(path, sep=sep))
    occs = pd.concat(occs, sort=False)
    # grab all the image files (careful, really slow!)
    #for root, dirs, files in os.walk('python/Lib/email'):
    #print len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
    img_paths = glob.glob(img_path)

    # how many images are missing?
    num_missed = len(occs) - len(img_paths)
    print("number of missing files: {}".format(img_paths))

    # get ids of files that are present
    img_ids = [ path.split("_alti")[0].split("/")[-1] for path in img_paths]

    # grab ids from both train and test set
    occ_ids = occs['id']

    # get the ids that are missing from the image dataset
    missing = us_cat_ids[~occ_ids.isin(img_ids)]

    # build a set of all the directories that are missing in the data
    missing_folders = set()
    for miss in us_missing:
        cdd, ab, cd = id_2_file(miss)
        subpath = "patches_us_{}".format(cdd) if id_ >= 10000000 else "patches_{}/{}".format('fr', cd)
        missing_folders.add(subpath)
    return missing_folders

def check_corrupted_files(base_dir):
   raise NotImplementedError 


def key_for_value(d, value):
    # this will be useful for final implementation
    return(list(d.keys())[list(d.values()).index(value)])

def normalize_latlon(latlon, min_, scale):
    return (latlon - min_)/scale