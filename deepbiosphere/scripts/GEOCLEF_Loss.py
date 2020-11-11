import os
import numpy as np
import operator
from functools import reduce
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# trick is going to be figuring out the presence-only calculations

# Brier loss using sigmoid with only present classes

def pass_(input):
    return input

class BrierPresenceOnly(nn.Module):
    def __init__(self, type='sum'):
        #include sigmoid or softmax option
        super().__init__()
        self.op = None
        self.type = type
        if type == 'mean':
            self.op = torch.mean
        elif type == 'sum':
            self.op = torch.sum
        elif type == 'none':
            self.op = pass_
        else:
            raise NotImplementedError
            
    # taken from https://discuss.pytorch.org/t/how-to-calculate-accuracy-multi-label/73883/3
    # targets will be a multi-hot vector (batched)    
    def forward(self, inputs, targets, smooth=1):        
        
        inputs = F.softmax(inputs)
        sub = (targets - inputs)
        # will return to presence-only, because if target
        # has 0, multiplication will zero it out
        res = sub ** 2
        res = res * targets
        # average can only be across positive classes
        res_avg = torch.div(torch.sum(res, dim=1), targets.sum(dim=1))
        return self.op(res_avg)

# Brier loss using all classes
class BrierAll(nn.Module):
    def __init__(self, type='sum'):
        super().__init__()
        self.type = type
        if type == 'mean':
            self.op = torch.mean
        elif type == 'sum':
            self.op = torch.sum
        elif type == 'none':
            self.op = pass_
        else:
            raise NotImplementedError

    def forward(self, inputs, targets, smooth=1):
        inputs = F.sigmoid(inputs)
        sub = (targets - inputs)
        res = sub**2
        res = torch.mean(res, dim=1)
        # call mse_loss here TODO
        
        return self.op(res)
    
# presence-only cross entropy
# from https://gist.github.com/mjdietzx/50d3c26f1fd543f1808ffffacc987cbf
# https://jamesmccaffrey.wordpress.com/2020/03/16/the-math-derivation-of-the-softmax-with-max-and-log-tricks/
class CrossEntropyPresenceOnly(nn.Module):
    # for now, gotta have weight so just pass 1s for laziness
    def __init__(self, class_weights,  device='cpu', type='sum'):
        super().__init__()
        self.type = type
        if type == 'mean':
            self.op = torch.mean
        elif type == 'sum':
            self.op = torch.sum
        elif type == 'none':
            self.op = pass_
        else:
            raise NotImplementedError
        self.class_weights = torch.autograd.Variable(class_weights) 
        self.log_softmax = nn.LogSoftmax()    

    # targets is a multihot vector
    def forward(self, inputs, targets):

        log_probabilities = self.log_softmax(inputs)
        # NLLLoss(x, class) = -weights[class] * x[class]
        nll = -targets * log_probabilities
        nll_weight = self.class_weights * nll
        # now average across sample
        nll_avg = torch.div(torch.sum(nll, dim=1), targets.sum(dim=1))
        # then average across batch
        return self.op(nll_avg)

    
    
    
    # from https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
class AsymmetricLoss(nn.Module):
    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

    def forward(self, x, y):
        """"
        Parameters
        ----------
        x: input logits
        y: targets (multi-label binarized vector)
        """

        # Calculating Probabilities
        x_sigmoid = torch.sigmoid(x)
        xs_pos = x_sigmoid
        xs_neg = 1 - x_sigmoid

        # Asymmetric Clipping
        if self.clip is not None and self.clip > 0:
            xs_neg = (xs_neg + self.clip).clamp(max=1)

        # Basic CE calculation
        los_pos = y * torch.log(xs_pos.clamp(min=self.eps))
        los_neg = (1 - y) * torch.log(xs_neg.clamp(min=self.eps))
        loss = los_pos + los_neg

        # Asymmetric Focusing
        if self.gamma_neg > 0 or self.gamma_pos > 0:
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            pt0 = xs_pos * y
            pt1 = xs_neg * (1 - y)  # pt = p if t > 0 else 1-p
            pt = pt0 + pt1
            one_sided_gamma = self.gamma_pos * y + self.gamma_neg * (1 - y)
            one_sided_w = torch.pow(1 - pt, one_sided_gamma)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            loss *= one_sided_w

        return -loss.sum()

# from https://github.com/Alibaba-MIIL/ASL/blob/main/src/loss_functions/losses.py
class AsymmetricLossOptimized(nn.Module):
    ''' Notice - optimized version, minimizes memory allocation and gpu uploading,
    favors inplace operations'''

    def __init__(self, gamma_neg=4, gamma_pos=1, clip=0.05, eps=1e-8, disable_torch_grad_focal_loss=False):
        super().__init__()

        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip
        self.disable_torch_grad_focal_loss = disable_torch_grad_focal_loss
        self.eps = eps

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
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(False)
            self.xs_pos = self.xs_pos * self.targets
            self.xs_neg = self.xs_neg * self.anti_targets
            self.asymmetric_w = torch.pow(1 - self.xs_pos - self.xs_neg,
                                          self.gamma_pos * self.targets + self.gamma_neg * self.anti_targets)
            if self.disable_torch_grad_focal_loss:
                torch._C.set_grad_enabled(True)
            self.loss *= self.asymmetric_w

        return -self.loss.sum()
    
    
    