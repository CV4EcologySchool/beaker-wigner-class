# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 15:03:09 2022
util file i guess
@author: tnsak
"""
import random
import torch
from torch.backends import cudnn
import torch.nn as nn
import numpy as np

def init_seed(seed):
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        cudnn.benchmark = True
        cudnn.deterministic = True

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction: str = 'mean'):
        super().__init__()
        if reduction not in ['mean', 'none', 'sum']:
            raise NotImplementedError('Reduction {} not implemented.'.format(reduction))
        self.reduction = reduction
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, x, target):
        eps = np.finfo(float).eps
        p_t = torch.where(target == 1, x, 1-x)
        fl = - 1 * (1 - p_t) ** self.gamma * torch.log(p_t + eps)
        fl = torch.where(target == 1, fl * self.alpha, fl * (1 - self.alpha))
        return self._reduce(fl)

    def _reduce(self, x):
        if self.reduction == 'mean':
            return x.mean()
        elif self.reduction == 'sum':
            return x.sum()
        else:
            return x

class CrossEntSNR(nn.CrossEntropyLoss):
    def __init__(self):
        super().__init__()
    
    def forward(self, x, target, snr_scale):
        ce = super().forward(x, target)
        ce = ce * snr_scale
        return ce.mean()
