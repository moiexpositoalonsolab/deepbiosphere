from types import SimpleNamespace
import torch
import json

## ---------- Paths to important directories ---------- ##

paths = {
    'OCCS' : '/add/your/directory/here/occurrences/',
    'SHPFILE' : '/add/your/directory/here/shpfiles/',
    'MODELS' : '/add/your/directory/here/models/',
    'IMAGES' : '/add/your/directory/here/images/',
    'RASTERS' : '/add/your/directory/here/rasters/',
    'BASELINES' : '/add/your/directory/here/baselines/',
    'RESULTS' : '/add/your/directory/here/results/',
    'SCRATCH' : "/add/your/directory/here/",
    'BLOB_ROOT' :  'https://naipeuwest.blob.core.windows.net/naip/' # have to usethe slow european image, us image got removed finally 'https://naipblobs.blob.core.windows.net/', #
}
paths = SimpleNamespace(**paths)

## ---------- File loading ---------- ##

def setup_pretrained_dirs():
    if not os.path.exists(f"{paths.MODELS}/pretrained/"):
        os.makedirs(f"{paths.MODELS}/pretrained/")
    return f"{paths.MODELS}/pretrained/"


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


# unfair for observations with >30 species, but whatever
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
    return perob / len(ytrue)
    
    
def species_topK(ytrue, yobs, K):
    yobs = torch.as_tensor(yobs)
    ytrue = torch.as_tensor(ytrue)
    # convert to probabilities if not done already
    if (yobs.min() <= 0.0) or (yobs.max() >= 1.0):
        yobs = torch.sigmoid(yobs)
    # convert 
    tk = torch.topk(yobs, K)
    # get all unique species label and their indices
    # the order is backward somehow TOOD
    unq = torch.unique(ytrue, sorted=False, return_inverse=True)
    # make a dict to store the results for each species
    specs = {v.item():[] for v in unq[0]}
    # go through each row and assign it to the corresponding
    # species using the reverse_index item from torch.unique
    for val, row in zip(unq[1], tk[1]):
        specs[unq[0][val.item()].item()].append(row)
    sas = []
    for spec, i in specs.items():
        # spoof ytrue for this species
        yt = torch.full((len(i),K), spec)
        # and calculate 'per-obs' accuracy
        sas.append((torch.stack(i)== yt).sum().item()/len(i))
    # and take average
    return sum(sas)/len(ytrue)

# taken from torchmetrics
# code taken from https://github.com/pytorch/tnt/pull/21/files/d6f1f0065cade3e2f8104049ba08fcf6d85d15c8
# explanation: https://blog.paperspace.com/mean-average-precision/
def mean_average_precision(scores, targets):
    scores = torch.as_tensor(scores)
    targets = torch.as_tensor(targets)
    if scores.numel() == 0:
        return 0
    ap = torch.zeros(scores.shape[1])
    rg = torch.arange(1, scores.shape[0]+1).float()
    # compute average precision for each class
    for k in range(scores.shape[1]):
        # sort scores
        currsc = scores[:, k]
        currtarg = targets[:, k]
        # ignore classes with no presence in set
        if currtarg.sum() == 0:
            ap[k] = np.nan
        else:
            _, sortind = torch.sort(currsc, 0, True)
            truth = currtarg[sortind]
            # compute true positive sums
            tp = truth.float().cumsum(0)

            # compute precision curve
            precision = tp.div(rg)
            # compute average precision
            ap[k] = precision[truth.bool()].sum() / max(truth.sum(), 1)
    # ignore absent classes
    return ap[~torch.isnan(ap)].mean().item()
