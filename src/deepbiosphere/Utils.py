import json
import torch
import numpy as np
from tqdm import tqdm
import sklearn.metrics as mets
from types import SimpleNamespace

## ---------- MAGIC NUMBERS ---------- ##

# TODO: change all magic numbers to this
# TODO: figure out what to do with this
# because I can't have it in Dtaset.py
# or nAIP_Utils.py because then they both
# recursively use the other. Might need
# to chnage all magic values to be in this file
# idk
IMG_SIZE = 256



## ---------- Paths to important directories ---------- ##

paths = {
    'OCCS' : '/home/lgillespie/deepbiosphere/data/occurrences/',
    'SHPFILES' : '/home/lgillespie/deepbiosphere/data/shpfiles/',
    'MODELS' : '/home/lgillespie/deepbiosphere/models/',
    'IMAGES' : '/home/lgillespie/deepbiosphere/data/images/',
    'RASTERS' : '/home/lgillespie/deepbiosphere/data/rasters/',
    'BASELINES' : '/home/lgillespie/deepbiosphere/data/baselines/',
    'RESULTS' : '/home/lgillespie/deepbiosphere/data/results/',
    'MISC' : '/home/lgillespie/deepbiosphere/data/misc/',
    'DOCS' : '/home/lgillespie/deepbiosphere/docs/',
    'SCRATCH' : "/NOBACKUP/scratch/lgillespie/naip/",
    'BLOB_ROOT' :  'https://naipeuwest.blob.core.windows.net/naip/' # have to usethe slow european image, us image got removed finally 'https://naipblobs.blob.core.windows.net/', #
}
paths = SimpleNamespace(**paths)

## ---------- File loading ---------- ##

def setup_pretrained_dirs():
    if not os.path.exists(f"{paths.MODELS}/pretrained/"):
        os.makedirs(f"{paths.MODELS}/pretrained/")
    return f"{paths.MODELS}/pretrained/"

## ---------- Data manipulation ---------- ##

# https://stackoverflow.com/questions/2659900/slicing-a-list-into-n-nearly-equal-length-partitions
def partition(lst, n):
    division = len(lst) / n
    return [lst[round(division * i):round(division * (i + 1))] for i in range(n)]

# https://stackoverflow.com/questions/312443/how-do-you-split-a-list-into-evenly-sized-chunks
def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def dict_key_2_index(df, key):
    return {
        k:v for k, v in
        zip(df[key].unique(), np.arange(len(df[key].unique())))
    }

# more clear version of uniform scaling
# https://stackoverflow.com/questions/5294955/how-to-scale-down-a-range-of-numbers-with-a-known-min-and-max-value
def scale(x, min_=None, max_=None, out_range=(-1,1)):

    if min_ == None and max_ == None:
        min_, max_ = np.min(x), np.max(x)
    return ((out_range[1]-out_range[0])*(x-min_))/(max_-min_)+out_range[0]

## ---------- Accuracy metrics ---------- ##

# calculates intersection of two tensors
# assumed y_t is already in pres/abs form
def pres_intersect(ob_t, y_t):
    # only where both have the same species keep
    sum_ = ob_t + y_t
    int_ = sum_ > 1
    return torch.sum(int_, dim=1)

# calculates union of two tensors
def pres_union(ob_t, y_t):
    sum_ = ob_t + y_t
    sum_ = sum_ > 0
    return torch.sum(sum_, dim=1)

def precision_per_obs(ob_t: torch.tensor, y_t: torch.tensor, threshold=.5):
    ob_t = torch.as_tensor(ob_t)
    y_t = torch.as_tensor(y_t)
    # threshold
    ob_t = (ob_t >= threshold).float()
    # get intersection
    top = pres_intersect(ob_t, y_t).float()
    # get # predicted species
    bottom = torch.sum(ob_t, dim=1)
    # divide!
    ans = torch.div(top, bottom)
    # if nan (divide by 0) just set to 0.0
    ans[ans != ans] = 0
    return ans

def recall_per_obs(ob_t, y_t, threshold=.5):
    ob_t = torch.as_tensor(ob_t)
    y_t = torch.as_tensor(y_t)
    # threshold
    ob_t = (ob_t >= threshold).float()
    # get intersection
    top = pres_intersect(ob_t, y_t).float()
    # get # observed species
    bottom = torch.sum(y_t, dim=1)
    ans = torch.div(top, bottom)
    # if nan (divide by 0) just set to 0.0
    # this relies on the assumption that all nans are 0-division
    ans[ans != ans] = 0
    return ans

def accuracy_per_obs(ob_t, y_t, threshold=.5):
    ob_t = torch.as_tensor(ob_t)
    y_t = torch.as_tensor(y_t)
    # threshold
    ob_t = (ob_t >= threshold).float()
    # intersection
    top = pres_intersect(ob_t, y_t).float()
    # union
    bottom = pres_union(ob_t, y_t)
    ans = torch.div(top, bottom)
    # if nan (divide by 0) just set to 0.0
    # this relies on the assumption that all nans are 0-division
    ans[ans != ans] = 0
    return ans

def f1_per_obs(ob_t, y_t, threshold=.5):
    pre = precision_per_obs(ob_t, y_t, threshold)
    rec = recall_per_obs(ob_t, y_t, threshold)
    ans =  2*(pre*rec)/(pre+rec)
    # if denom=0, F1 is 0
    ans[ans != ans] = 0.0
    return ans
    
def obs_topK(ytrue, yobs, K):
    # ytrue should be spec_id, not all_specs_id
    yobs = torch.as_tensor(yobs)
    ytrue = torch.as_tensor(ytrue)
    # convert to probabilities if not done already
    if (yobs.min() <= 0.0) or (yobs.max() >= 1.0):
        yobs = torch.sigmoid(yobs)
    # convert
    tk = torch.topk(yobs, K)
    # compare indices and will be 1 for every row where
    # yob is in topK, 0 else. Summing across all dimensions
    # gives you final sum (since only 1 obs per row)
    perob = (tk[1]== ytrue.unsqueeze(1).repeat(1,K)).sum().item()
    # don't forget to average
    return (perob / len(ytrue)), (tk[1]== ytrue.unsqueeze(1).repeat(1,K))

def species_topK(ytrue, yobs, K):
    assert yobs.shape[1] > 1, "predictions are not multilabel!"
    nspecs = yobs.shape[1]
    yobs = torch.as_tensor(yobs)
    ytrue = torch.as_tensor(ytrue)
    # convert to probabilities if not done already
    if (yobs.min() <= 0.0) or (yobs.max() >= 1.0):
        yobs = torch.sigmoid(yobs)
    # convert
    tk = torch.topk(yobs, K)
    # get all unique species label and their indices
    unq = torch.unique(ytrue, sorted=False, return_inverse=True)
    # make a dict to store the results for each species
    specs = {v.item():[] for v in unq[0]}
    # go through each row and assign it to the corresponding
    # species using the reverse_index item from torch.unique
    for val, row in zip(unq[1], tk[1]):
        specs[unq[0][val.item()].item()].append(row)
    sas = []
    # add every species so csv writing works
    for i in range(nspecs):
        # ignore not present species
        spec = specs.get(i)
        if spec is None:
            sas.append(np.nan)
            continue
        nspecs += 1
        # spoof ytrue for this species
        yt = torch.full((len(spec),K), i)
        # and calculate 'per-obs' accuracy
        sas.append((torch.stack(spec)== yt).sum().item()/len(spec))
    sas = np.array(sas)
    gsas = sas[~np.isnan(sas)]
    sas = np.array(sas)
    gsas = sas[~np.isnan(sas)]
    return (sum(gsas) / len(gsas)), sas


def mean_calibrated_roc_auc_prc_auc(y_true, y_obs, npoints=50):
    assert y_true.shape == y_obs.shape
    assert y_true.shape[1] > 1
    assert y_obs.shape[1] > 1
    roc_auc, prc_auc = [],[]
    for i in tqdm(range(y_obs.shape[1]), total=y_obs.shape[1], unit='species'):
        ra,pa = calibrated_roc_auc_prc_auc(y_true[:,i], y_obs[:,i])
        roc_auc.append(ra)
        prc_auc.append(pa)
    return roc_auc, prc_auc

# precision: tp / (tp + fp)
# recall, TPR: tp / (tp + fn)
# FPR = FP/(FP+TN)
def calibrated_roc_auc_prc_auc(y_true, y_obs, npoints=50):
    cutoffs = np.linspace(0.0, 1.0, npoints)
    tpr, fpr, pre = [],[],[]
    # ignore when there is no actual present case of this species
    if y_true.sum() == 0:
        return (np.nan, np.nan)
    for i in cutoffs:
        pred = (y_obs >= i).astype(np.short)
        tn, fp, fn, tp = mets.confusion_matrix(y_true, pred).ravel()
        if (tp + fn) > 0:
            tpr.append(tp / (tp + fn))
        else:
            tpr.append(0)
        if (fp+tn) > 0:
            fpr.append(fp/(fp+tn))
        else:
            fpr.append(0)
        if (tp + fp) > 0:
            pre.append(tp / (tp + fp))
        else:
            pre.append(0)
    # precision-recall x=recall, y=precision
    # roc: x= fpr, y= tpr
    return (mets.auc(tpr, pre),  mets.auc(fpr, tpr))