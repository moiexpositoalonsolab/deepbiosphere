import os
import numpy as np
import math
import pandas as pd
import glob
import torch
import matplotlib.pyplot as plt
# adding this to check github integration on slack
# TODO: move general methods into here

def dict_key_2_index(df, key):
    return {
        k:v for k, v in 
        zip(df[key].unique(), np.arange(len(df[key].unique())))
    }



def dict_from_columns(df, key_col, val_col):
    ret = {}
    assert key_col in df.columns and val_col in df.columns
    for key, val in zip(df[key_col], df[val_col]):
        ret[key] = val
    return ret

# https://www.movable-type.co.uk/scripts/latlong.html
def nmea_2_meters(lat1, lon1, lat2, lon2):
    
    R = 6371009 #; // metres
    r1 = lat1 * math.pi/180 #; // φ, λ in radians
    r2 = lat2 * math.pi/180;
    dr = (lat2-lat1) * math.pi/180;
    dl = (lon2-lon1) * math.pi/180;

    a = math.sin(dr/2) * math.sin(dr/2) + \
              math.cos(r1) * math.cos(r2) * \
              math.sin(dl/2) * math.sin(dl/2);
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a));

    d = R * c #; // in metres
    return d


def add_taxon_metadata(base_dir, obs, organism):
    
    ## getting family, genus, species ids for each observation
    # get all relevant files
    print("adding taxon information")   
    gbif_meta = pd.read_csv("{}occurrences/species_metadata.csv".format(base_dir), sep=None)    
    present_specs = obs.species_id.unique()    
    # get all the gbif species ids for all the species in the us sample
    conversion = gbif_meta[gbif_meta['species_id'].isin(present_specs)]
    gbif_specs = conversion.GBIF_species_id.unique()
    # get dict that maps CELF id to GBIF id
    spec_2_gbif = dict(zip(conversion.species_id, conversion.GBIF_species_id))
    obs['gbif_id'] = obs['species_id'].map(spec_2_gbif)    
    taxons = pd.read_csv("{}occurrences/Taxon.tsv".format(base_dir), sep="\t")
    taxa = taxons[taxons['taxonID'].isin(gbif_specs)]
    phylogeny = taxa[['taxonID', 'kingdom', 'phylum', 'class', 'order', 'family', 'genus', 'canonicalName']]
    gbif_2_king = dict(zip(phylogeny.taxonID, phylogeny.kingdom))
    gbif_2_phy = dict(zip(phylogeny.taxonID, phylogeny.phylum))
    gbif_2_class = dict(zip(phylogeny.taxonID, phylogeny['class']))
    gbif_2_ord = dict(zip(phylogeny.taxonID, phylogeny.order))
    gbif_2_fam = dict(zip(phylogeny.taxonID, phylogeny.family))
    gbif_2_gen = dict(zip(phylogeny.taxonID, phylogeny.genus))
    gbif_2_spec = dict(zip(phylogeny.taxonID, phylogeny.canonicalName))    
    obs['family'] = obs['gbif_id'].map(gbif_2_fam)
    obs['genus'] = obs['gbif_id'].map(gbif_2_gen)
    obs['order'] = obs['gbif_id'].map(gbif_2_ord)
    obs['class'] = obs['gbif_id'].map(gbif_2_class)
    obs['phylum'] = obs['gbif_id'].map(gbif_2_phy)
    obs['kingdom'] = obs['gbif_id'].map(gbif_2_king)
    obs['species'] = obs['gbif_id'].map(gbif_2_spec)
    if organism == 'plant':
        obs = obs[obs.kingdom == 'Plantae']
    elif organism == 'animal':
        obs = obs[obs.kingdom == 'Animalia']
    else: #plantanimal
        pass
    return obs

def cnn_output_size(in_size, kernel_size, stride, padding):
    output = int((in_size - kernel_size + 2*(padding)) / stride) + 1
    return(output)

def torch_scale(x, out_range=(0, 1), min_=None, max_=None):

    if min_ == None and max_ == None:
        min_, max_ = x.min(), x.max()
        # .float() converts to float32, we want float64 for now...
        # jk, we want float32 for data efficiency reasons
    min_, max_ = min_.to(dtype=torch.float32, device=x.device), max_.to(dtype=torch.float32, device=x.device)
    x = x.to(dtype=torch.float32, device=x.device)
    y = (x - (max_ + min_) / 2) / (max_ - min_)
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def scale(x, out_range=(-1, 1), min_=None, max_=None):
    
    if min_ == None and max_ == None:
        min_, max_ = np.min(x), np.max(x)
    y = (x - (max_ + min_) / 2) / (max_ - min_)
    return y * (out_range[1] - out_range[0]) + (out_range[1] + out_range[0]) / 2

def plot_image(base_dir, id_, figsize=(10,10), transpose=False, altitude=True):
    imgs =   image_from_id(id_, base_dir, altitude) 
    print(imgs.shape, imgs.shape[0]-2)
    num = imgs.shape[0]-2
    fig, axs = plt.subplots(num, figsize=figsize) if transpose else plt.subplots(1, num, figsize=figsize)
    if altitude:
        axs[0].imshow(np.transpose(imgs[1:4,:,:], [1,2,0]))
        axs[0].set_title("rgb ")
        axs[1].imshow(imgs[4:5,:,:].squeeze())
        axs[1].set_title("infrared")
        axs[2].imshow(imgs[0,:,:].squeeze())
        axs[2].set_title("altitude")
    else:
# imgs.take([0,2,3], mode='wrap', axis=0).shape
        axs[0].imshow(np.transpose(imgs[:3]).squeeze())
        axs[0].set_title("rgb ")
        axs[1].imshow(imgs[3,:,:].squeeze())
        axs[1].set_title("infrared")
    return fig

def plot_env_rasters(rasters, figsize=(10,10)):
    fig, axs = plt.subplots(5,5, figsize=figsize)
    for ras, i in zip(axs.ravel(), range(len(rasters))):
        ras.imshow(rasters[i,:,:].squeeze())
    return fig

def plot_one_env_rasters(rasters, idx=0, figsize=(10,10)):
    fig, axs = plt.subplots(1, figsize=figsize)
    axs.imshow(rasters[idx,:,:].squeeze())
    return fig


def norm_rank(curr, stop):
    if curr == stop:
        return 1/stop
    else:
        ncur = curr +1
        return norm_rank(ncur, stop) + 1/curr

def mean_reciprocal_rank(lab, guess, total=True, norm=True):  
    
    # sort 
    pred, idxs = torch.topk(guess, guess.shape[-1])
    pnp = idxs.tolist()[0]
    all_right = []

    for i in range(len(lab)):
        all_right.append(pnp.index(lab[i]))
    all_right = np.array(all_right)
    all_right += 1
    if total:
        if norm:
            normed = normed_rank[len(lab)]
            return ((1 / all_right).sum()) / normed
        else:
            
            return (1 / all_right).sum()

    else:
        # just want the first correct index reciprocal rank
        if norm:
            normed = normed_rank[len(lab)]
            return (1/all_right.min())/normed
        else:
            return 1/all_right.min()    
        
def proba_intersect(ob_t, y_t, device):
    
    _, indxs_pred = torch.sort(ob_t, dim=1, descending = True)
    to_get, indxs_pres = torch.sort(y_t, dim=1, descending = True)
    mask = (to_get == 0.0)

    pres = indxs_pres.masked_fill(mask, -1)
    pred = indxs_pred.masked_fill(mask, -1)

    inter = torch.full([pres.shape[0]], -1, device=device)
    for i, (m, n) in enumerate(zip(pres, pred)):
        mask1 = m >=0
        a = torch.masked_select(m, mask1)
        b = torch.masked_select(n, mask1)    
        c = torch_intersection(a, b, device)
        inter[i] = len(c)
    del indxs_pred, to_get, indx_pred, mask, pres, pred
    return inter

# assumed y_t is already in pres/abs form
def pres_intersect(ob_t, y_t):    
    # only where both have the same species keep
    sum = ob_t + y_t
    int = (sum > 1)
    del sum
    return torch.sum(int, dim=1)


def precision_per_obs(ob_t, y_t, device=torch.device("cpu"), threshold=.5, proba=False, alternate=False):
# proba is presence absence vs proba data in obs_t and alernate is which interesection to use
    
    if proba:
        if alternate:
            top = proba_intersect(ob_t, y_t, device)
#             ob_t = torch.sigmoid(ob_t)
            # then threshold probability
            ob_t = (ob_t > threshold).float()   
            # something squirrely here
            
            bottom = torch.sum(ob_t, dim=1)            
        else:
#             ob_t = torch.sigmoid(ob_t)
            # then threshold probability
            ob_t = (ob_t > threshold).float()
#             print(ob_t.sum(dim=1), y_t.sum(dim=1))
            top = pres_intersect(ob_t, y_t).float()
            bottom = torch.sum(ob_t, dim=1)
    else: 
            # needto threshold probas
#         print(ob_t.sum(dim=1), y_t.sum(dim=1))            
        if alternate:
            raise NotImplementedError("can't do probability thresholding on presence only data!")
        top = pres_intersect(ob_t, y_t).float()
        bottom = torch.sum(ob_t, dim=1)
#     print(top.dtype, bottom.dtype)
    ans = torch.div(top, bottom)
    # this relies on the assumption that all nans are 0-division
    ans[ans != ans] = 0
    return ans


def recall_per_obs(ob_t, y_t, device=torch.device("cpu"), threshold=.5, proba=False, alternate=False):
# proba is presence absence vs proba data in obs_t and alernate is which interesection to use
    
    if proba:
        if alternate:
            top = proba_intersect(ob_t, y_t, device)            
        else:
#             ob_t = torch.sigmoid(ob_t)
            # then threshold probability
            ob_t = (ob_t > threshold).float()
#             print(ob_t.sum(dim=1), y_t.sum(dim=1))
            top = pres_intersect(ob_t, y_t).float()

    else: 
            # needto threshold probas
        if alternate:
            raise NotImplementedError("can't do probability thresholding on presence only data!")
        top = pres_intersect(ob_t, y_t).float()
 
    bottom = torch.sum(y_t, dim=1)
#     print(top, bottom)    
    ans = torch.div(top, bottom)
    # this relies on the assumption that all nans are 0-division
    ans[ans != ans] = 0
    return ans


def f1_per_obs(ob_t, y_t, device=torch.device("cpu"), threshold=.5, proba=False, alternate=False):
# proba is presence absence vs proba data in obs_t and alernate is which interesection to use
    
    
    if proba:
        if alternate:
            top = proba_intersect(ob_t, y_t, device)
#             ob_t = torch.sigmoid(ob_t)
            # then threshold probability
            ob_t = (ob_t > threshold).float()            
            bottom1 = torch.sum(ob_t, dim=1)            
        else:
#             ob_t = torch.sigmoid(ob_t)
            # then threshold probability
            ob_t = (ob_t > threshold).float()
            top = pres_intersect(ob_t, y_t).float()
            bottom1 = torch.sum(ob_t, dim=1)
    else: 
            # needto threshold probas
        if alternate:
            raise NotImplementedError("can't do probability thresholding on presence only data!")
        top = pres_intersect(ob_t, y_t).float()
        bottom1 = torch.sum(ob_t, dim=1)
       
    bottom2 = torch.sum(y_t, dim=1)
    # sum bottom1 and bottom2 along axis 1
    bottom = bottom1 + bottom2
    ans = torch.div(top, bottom)
    # this relies on the assumption that all nans are 0-division
    ans[ans != ans] = 0
    return ans

def recall_per_batch(output, target, actual):

    recall = []
    tot_rec = []
    top1_rec = []
    tot_top1 = []
    for obs, trg, act in zip(output, target, actual):
        out_vals, out_idxs = torch.topk(obs, int(trg.sum().item()))
        targ_vals, targ_idxs = torch.topk(trg, int(trg.sum().item()))
        eq = len(list(set(out_idxs.tolist()) & set(targ_idxs.tolist())))
        eq_t1 = len(list(set(out_idxs.tolist()) & set([actual.item()])))        
        recall = eq / trg.sum() * 100
        top1_recall = eq_t1 * 100
        tot_rec.append((eq, len(targ_idxs)))
        recall.append(recall.item())
        tot_top1.append(eq_t1)
        top1_rec.append(top1_recall.item())
    return recall, tot_rec, top1_rec, tot_top1
        
    

# not batched version!!
# t1 should model output, t2 should be label for batch dims to work
# https://stackoverflow.com/questions/55110047/finding-non-intersection-of-two-pytorch-tensors
def torch_intersection(t1, t2, device):
    
    indices = torch.zeros_like(t2, dtype = torch.uint8, device = device)
    if len(t2.shape) > 1:
        raise TypeError; 'will index incorrectly across tensors'
    for i, elem in zip(np.arange(len(t2)), t2):
        indices[i] = (t1 == elem).sum()
    intersection = t2[indices]  
    return intersection

def recall_per_example_classes(lab, guess, actual):
    """ calculates recall per example. Returns multi-label recall + top 1 recall + all correctly predicted classes"""
    # recall
    maxk = len(lab)
    pred, idxs = torch.topk(guess, maxk)
    corr_id = list(set(idxs.tolist()[0]) & set(lab)) 
    eq = len(corr_id)
    recall = eq / maxk
    top1_recall = len(list(set(idxs.tolist()[0]) & set([actual])))
    return recall, top1_recall, corr_id
    

# so sigmoiding doesn't really change the output, getting rid of it
def recall_per_example(lab, guess, actual, weight, device='cpu'):

    maxk = len(lab)
    pred, idxs = torch.topk(guess, maxk)
    intersection = torch_intersection(idxs, lab, device)
    recall = len(intersection)/maxk
    top1_rec = (intersection == actual).sum()
    top1_rec_weight = float(top1_rec)/float(weight)
    return recall, top1_rec.tolist(), top1_rec_weight, intersection.tolist()


# so sigmoiding doesn't really change the output, getting rid of it
def accuracy_per_example(lab, guess, device='cpu'):
    
    guess = torch.sigmoid(guess)
    yhat_size = (guess > .5).sum()
    pred, idxs = torch.topk(guess, yhat_size)
    
    intersection = torch_intersection(idxs, lab, device)
    union = torch.unique(torch.cat([idxs[0], lab]))
    return float(len(intersection))/len(union), intersection.tolist(), union.tolist()

# NEW threshold is .5 and we will pass the values through sigmoid to convert the distro
# to a proability for each class
def precision_per_example(lab, guess, actual, weight, device='cpu', thres=.5):
    
    guess = torch.sigmoid(guess)
    yhat_size = (guess > .5).sum()
    pred, idxs = torch.topk(guess, yhat_size)
    overlap = torch_intersection(idxs, lab, device)
    prec = float(len(overlap))/ float(yhat_size) if yhat_size > 0 else 0.0
    top1_prec = (overlap == actual).sum()
    top1_prec_weight = float(top1_prec)/float(weight)    
    return prec, top1_prec.tolist(), top1_prec_weight, overlap.tolist()

def num_corr_matches(output, target, actual):
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
    _, pred = torch.topk(output, maxk, 1, True, True)
    _, targ_idx = torch.topk(target, 1, 1, True, True)
    targ = targ_idx[0].unsqueeze(1).repeat(1,maxk).to(device)
    correct = pred.eq(targ)
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0).cpu()
        res.append(correct_k.mul_(100.0 / batch_size))
    del targ, pred, target
    return res
def subpath_2_img_noalt(pth, subpath, id_):
    rgbd = "{}{}{}.npy".format(pth, subpath, id_)    
    # Necessary because some data corrupted...
    np_img = np.load(rgbd)
    np_img = np_img[:,:,:4]
    x = np.transpose(np_img,(2, 0, 1))
    return x

def image_from_id(id_, pth, means, stds, altitude=False, sub_mean="none"):
    # make sure image and path are for same region
    cdd, ab, cd = id_2_file(id_)
    subpath = "patches_{}/{}/{}/".format('fr', cd, ab) if id_ >= 10000000 else "patches_{}/patches_{}_{}/{}/{}/".format('us', 'us', cdd, cd, ab)
    img = subpath_2_img(pth, subpath, id_) if altitude else subpath_2_img_noalt(pth, subpath, id_)
    # decision: always 0-1 scale images
    # supposed to be numpy array at this point still, so use scale not tensor_scale
    img = scale(img, out_range=(0,1), min_=0, max_=255)
    # can use copy since not obj array https://numpy.org/doc/stable/reference/generated/numpy.copy.html
    imgg = np.copy(img)
    # if unecessary if sub_mean calculation doens't scale
    #if sub_mean:
    for i, (mean, std, channel) in enumerate(zip(means, stds, img)):

        # this is the transformation that was previously happening...
        # no it's not
#             n = (np.full( (256,256), 256) + (mean - channel)).round()
#             print("aha? ", n[:5,:5].round())
# How we should be normalizing: https://www.geeksforgeeks.org/how-to-normalize-images-in-pytorch/
    # what the hell is going on here???
    # so, not sure what the internal python transformation that was happening by changing the underlying data, so instead will just copy over
        #img[i,:,:] = mean - channel # this is incorrect, and is doing some wild ting?
        imgg[i,:,:] = (channel - mean)/std
    return imgg

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
    
def clean_all_models(base_dir, data='nets', num_2_keep=5):
    # https://stackoverflow.com/questions/16953842/using-os-walk-to-recursively-traverse-directories-in-python
    # traverse root directory, and list directories as dirs and files as files
    for root, dirs, files in os.walk("{}{}/".format(base_dir, data)):
        path = root.split(os.sep)
        print((len(path) - 1) * '---', os.path.basename(root))
        print(len(files))
#         print(root, dirs)
        # unique files are based on lr, e
        unq_runs= {file.split("_e")[0] for file in files}
        # so this is one entry per run
        # list of all epochs per run
        for run in unq_runs:
            pths = glob.glob(root+ "/"+ run + "_e*.tar")
            srted = sort_by_epoch(pths)
            to_keep = srted[-num_2_keep:]
            to_toss = srted[:-num_2_keep]
            if len(to_keep) > 0:
                print("missing models for {}".format(run))
                pass
            else:
                if len(to_toss) > 0:
                    print("removing epochs {} to {} and keeping epochs {} to {} of model {}".format(
                        strip_to_epoch([to_toss[0]])[0], 
                        strip_to_epoch([to_toss[-1]])[0],
                        strip_to_epoch([to_keep[0]])[0], 
                        strip_to_epoch([to_keep[-1]])[0],
                        path_to_cfgname(run)))
                    for to_remove in to_toss:
                        os.remove(to_remove)
        print("\n")
        
        
def numpy_2_df(np_arr, columns, other_df=None, cols_2_steal=None):
    df_arr = pd.DataFrame(np_arr, columns=columns)
    if other_df is not None:
        df_arr[cols_2_steal] = other_df[cols_2_steal]
    return df_arr
    
def sort_by_epoch(paths, reversed=False):    
    return sorted(paths, reverse=reversed,key= lambda x: (int(x.split('_e')[-1].split('.')[0])))

def strip_to_epoch(filepaths):
#     print(filepaths[0])
#     print(filepaths[0].split('.')[0])
    return  [int(f.split('_e')[-1].split('.')[0]) for f in filepaths] 

def path_to_cfgname(filepath):
    return filepath.split("/")[-1].split("_e")[0].split('.')[0]
    
'''files are default assumed to be ';' separator '''
def check_gbif_files(occ_paths, img_path, sep=';'):
    occs = []
    for path in occ_paths:
        occs.append(pd.read_csv(path, sep=None))
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


def check_batch_size(observation, dataset, processes, loader, batch_size, optimizer, net, spec_loss, fam_loss, gen_loss, sampler, device):
    if batch_size < 1:
        exit(1), "non-memory error present!"
    for ret in loader:
        if observation == 'single':
            labels, batch = ret
            specs_lab = labels[:,0]
            gens_lab = labels[:,1]
            fams_lab = labels[:,2]
        elif observation == 'joint':
            specs_lab, gens_lab, fams_lab, batch = ret
        try:
            forward_one_example(specs_lab, gens_lab, fams_lab, batch, optimizer, net, spec_loss, gen_loss, fam_loss, device)
        except RuntimeError:
            batch_size -= 1
            print("decreasing batch size to {}".format(batch_size))
            loader = setup_dataloader(dataset, observation, batch_size, processes, sampler)
            check_batch_size(observation, dataset, processes, loader, batch_size, optimizer, net, spec_loss, fam_loss, gen_loss, sampler, device)
        break
    return batch_size, loader
    
