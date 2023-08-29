# torch functions
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# misc functions
import os
import operator
import numpy as np
from functools import reduce

import deepbiosphere.Utils as utils
from functools import reduce, partial


## ---------- softmax-based losses ---------- ##

# hacky way of getting pretty run code by
# dumping weighting in here
def CEWeighted(counts, id_dict, len_dset, device):
    # first convert species to ids
    pos_counts = {id_dict[sp]:val for sp, val in counts.items()}
    neg_counts ={ s: (len_dset-c) for s, c in counts.items()}
    # then, sort by id so it's in 0-S order
    pos_counts = sorted(pos_counts.items(), key= lambda x: x[0])
    neg_counts = sorted(neg_counts.items(), key= lambda x: x[0])
    _, pc = zip(*pos_counts)
    _, nc = zip(*neg_counts)
    # people recommend doing num_neg / num_pos to get the weighting....
    weights = [n/p for p,n in zip(pc, nc)]
    return nn.CrossEntropyLoss(weight=torch.tensor(weights,device=device))


# presence-only cross entropy
# from https://gist.github.com/mjdietzx/50d3c26f1fd543f1808ffffacc987cbf
# https://jamesmccaffrey.wordpress.com/2020/03/16/the-math-derivation-of-the-softmax-with-max-and-log-tricks/
# https://gombru.github.io/2018/05/23/cross_entropy_loss/
class CrossEntropyPresenceOnly(nn.Module):
    # for now, gotta have weight so just pass 1s for laziness
    def __init__(self, sum=False,class_weights=None,  weighting=False, device='cpu', reduce='sum'):
        super().__init__()
        self.type = type
        self.weighting = weighting
        if reduce == 'mean':
            self.op = torch.mean
        elif reduce == 'sum':
            self.op = torch.sum
        elif reduce == 'none':
            self.op = utils.pass_
        else:
            raise NotImplementedError
        if weighting:
            self.class_weights = torch.autograd.Variable(class_weights,device=device)
        self.log_softmax = nn.LogSoftmax(dim=1)

    # targets is a multihot vector
    def forward(self, inputs, targets):
        # wh yare we doing the log, and not just softmax??
        log_probabilities = self.log_softmax(inputs)
        # NLLLoss(x, class) = -weights[class] * x[class]
        nll = -targets * log_probabilities
	# handle class weighting
        if self.weighting:
            nll = self.class_weights * nll
        # now average across sample to correct for observations with more species
        # as done here: https://gombru.github.io/2018/05/23/cross_entropy_loss/
        nll_avg = torch.div(torch.sum(nll, dim=1), targets.sum(dim=1))
        # then average across batch
        return self.op(nll_avg)

## ---------- sigmoid-based losses ---------- ##

class BCE(nn.Module):
    ''' Identical to torch.nn.BCEWithLogitsLoss with reduction=sum'''
    def __init__(self, eps=1e-8, reduce='sum'):
        super().__init__()
        self.eps = eps
        if reduce == 'mean':
            self.op = torch.mean
        elif reduce == 'sum':
            self.op = torch.sum
        elif reduce == 'none':
            self.op = utils.pass_
        else:
            raise NotImplementedError
        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg  = self.loss = None
    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """
        self.targets = y
        self.anti_targets = 1 - y
        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos
        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))
        # return -self.loss.sum()
        return self.op(-self.loss)


# hacky way of getting pretty run code by
# dumping weighting in here
def BCEWeighted(counts, id_dict, len_dset, device):
    # first convert species to ids
    pos_counts = {id_dict[sp]:val for sp, val in counts.items()}
    neg_counts ={ s: (len_dset-c) for s, c in counts.items()}
    # then, sort by id so it's in 0-S order
    pos_counts = sorted(pos_counts.items(), key= lambda x: x[0])
    neg_counts = sorted(neg_counts.items(), key= lambda x: x[0])
    _, pc = zip(*pos_counts)
    _, nc = zip(*neg_counts)
    weights = [n/p for p,n in zip(pc, nc)]
    # https://stackoverflow.com/questions/66660354/understanding-pos-weight-argument-in-bcewithlogitsloss
    #  so in BCEWLL, pos_weight > 1 means that class being positive contributes more to the loss
    # now in many posts, https://discuss.pytorch.org/t/weights-in-bcewithlogitsloss/27452/8
    # people recommend doing num_neg / num_pos to get the weighting....
    return nn.BCEWithLogitsLoss(pos_weight=torch.tensor(weights,device=device))


class BCEScaled(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''
    def __init__(self, eps=1e-8, reduce='sum'):

        super().__init__()
        self.eps = eps
        if reduce == 'mean':
            self.op = torch.mean
        elif reduce == 'sum':
            self.op = torch.sum
        elif reduce == 'none':
            self.op = utils.pass_
        else:
            raise NotImplementedError
        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg  = self.loss = self.pos_loss = self.neg_loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Scaling by present and absent classes
        self.pos_loss = torch.div((self.targets * self.loss).sum(axis=1), (self.targets.sum(axis=1)+self.eps))
        self.neg_loss = torch.div((self.anti_targets * self.loss).sum(axis=1), (self.anti_targets.sum(axis=1)+self.eps))
        self.loss = self.pos_loss + self.neg_loss

        # return -self.loss.sum()
        return self.op(-self.loss)


class BCEProbScaled(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''
    def __init__(self, gamma_neg=1, gamma_pos=1, eps=1e-8, reduce='sum'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.eps = eps
        if reduce == 'mean':
            self.op = torch.mean
        elif reduce == 'sum':
            self.op = torch.sum
        elif reduce == 'none':
            self.op = utils.pass_
        else:
            raise NotImplementedError
        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            self.loss *= self.asymmetric_w
            # scale by number of pos, negative classes
        self.pos_loss = torch.div((self.targets * self.loss).sum(axis=1), (self.targets.sum(axis=1)+self.eps))
        self.neg_loss = torch.div((self.anti_targets * self.loss).sum(axis=1), (self.anti_targets.sum(axis=1)+self.eps))
        self.loss = self.pos_loss + self.neg_loss

        # return -self.loss.sum()
        return self.op(-self.loss)


class FocalLossScaled(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, eps=1e-8, reduce='sum'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.eps = eps
        if reduce == 'mean':
            self.op = torch.mean
        elif reduce == 'sum':
            self.op = torch.sum
        elif reduce == 'none':
            self.op = utils.pass_
        else:
            raise NotImplementedError
        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos


        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            self.loss *= self.asymmetric_w
            # scale by number of pos, negative classes
        self.pos_loss = torch.div((self.targets * self.loss).sum(axis=1), (self.targets.sum(axis=1)+self.eps))
        self.neg_loss = torch.div((self.anti_targets * self.loss).sum(axis=1), (self.anti_targets.sum(axis=1)+self.eps))
        self.loss = self.pos_loss + self.neg_loss

        # return -self.loss.sum()
        return self.op(-self.loss)


class BCEProbClipScaled(nn.Module):
    def __init__(self, gamma_neg=1, gamma_pos=1, clip=0.05, eps=1e-8, reduce='sum'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        if reduce == 'mean':
            self.op = torch.mean
        elif reduce == 'sum':
            self.op = torch.sum
        elif reduce == 'none':
            self.op = utils.pass_
        else:
            raise NotImplementedError
        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            self.loss *= self.asymmetric_w
            # scale by number of pos, negative classes
        self.pos_loss = torch.div((self.targets * self.loss).sum(axis=1), (self.targets.sum(axis=1)+self.eps))
        self.neg_loss = torch.div((self.anti_targets * self.loss).sum(axis=1), (self.anti_targets.sum(axis=1)+self.eps))
        self.loss = self.pos_loss + self.neg_loss

        # return -self.loss.sum()
        return self.op(-self.loss)


# from https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
# optimized version
class AsymmetricLoss(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''
    # standard gamma is 4-, 1+
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduce='sum'):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        if reduce == 'mean':
            self.op = torch.mean
        elif reduce == 'sum':
            self.op = torch.sum
        elif reduce == 'none':
            self.op = utils.pass_
        else:
            raise NotImplementedError
        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg, self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            self.loss *= self.asymmetric_w

        # return -self.loss.sum()
        return self.op(-self.loss)


# from https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
class AsymmetricLossScaled(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''
# standard is gamma-=4, gamma+=1, clip=0.05. gamma > 0 turns on scaling by predicted prob
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, reduce='sum'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.eps = eps
        if reduce == 'mean':
            self.op = torch.mean
        elif reduce == 'sum':
            self.op = torch.sum
        elif reduce == 'none':
            self.op = utils.pass_
        else:
            raise NotImplementedError
        # prevent memory allocation and gpu uploading every iteration, and encourages inplace operations
        self.targets = self.anti_targets = self.xs_pos = self.xs_neg = self.asymmetric_w = self.loss = None

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        self.targets = y
        self.anti_targets = 1 - y

        # Calculating Probabilities
        self.xs_pos = torch.sigmoid(x)
        self.xs_neg = 1.0 - self.xs_pos

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            self.xs_neg.add_(self.clip).clamp_(max=1)

        # Basic CE calculation
        self.loss = self.targets * torch.log(self.xs_pos.clamp(min=self.eps))
        self.loss.add_(self.anti_targets * torch.log(self.xs_neg.clamp(min=self.eps)))

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            self.loss *= self.asymmetric_w
            # scale by number of pos, negative classes
        self.pos_loss = torch.div((self.targets * self.loss).sum(axis=1), (self.targets.sum(axis=1)+self.eps))
        self.neg_loss = torch.div((self.anti_targets * self.loss).sum(axis=1), (self.anti_targets.sum(axis=1)+self.eps))
        self.loss = self.pos_loss + self.neg_loss

        # return -self.loss.sum()
        return self.op(-self.loss)



# ---------- Types ---------- #

# valid losses
class Loss(utils.FuncEnum, metaclass=utils.MetaEnum):

    CE                 = partial(nn.CrossEntropyLoss)
    WEIGHTED_CE        = partial(CEWeighted)
    CPO                = partial(CrossEntropyPresenceOnly)    # NOT tried
    BCE                = partial(BCE) # NOT tried
    WEIGHTED_BCE       = partial(BCEWeighted)
    SCALED_BCE         = partial(BCEScaled)  # NOT tried
    PROBSCALED_BCE     = partial(BCEProbScaled)  # NOT tried
    SCALED_FOCAL       = partial(FocalLossScaled)  # NOT tried
    PROBCLIPSCALED_BCE = partial(BCEProbClipScaled)  # NOT tried
    ASL                =  partial(AsymmetricLoss)  # NOT tried
    SCALED_ASL         = partial(AsymmetricLossScaled)  # NOT tried
