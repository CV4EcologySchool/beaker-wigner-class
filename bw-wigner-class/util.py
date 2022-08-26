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
import os
import matplotlib.pyplot as plt
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor, Normalize
from model import BeakerNet

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
    def __init__(self, weight):
        super().__init__(weight=weight)
    
    def forward(self, x, target, snr_scale):
        ce = super().forward(x, target)
        ce = ce * snr_scale
        return ce.mean()

# df is dataframe of predictions, model is .pt
def get_saliency(df, cfg, model):
    # load .pt file as BeakerNet
    if isinstance(model, str):
        state = torch.load(open(model, 'rb'), map_location='cpu')
        model = BeakerNet(cfg)
        model.load_state_dict(state['model'])
    
    model.eval()
    sal_out = []
    # loop through each row of DF and load file to predict on
    for i, v in enumerate(df.itertuples(index=False)):
        # loading image - 128x128 nparray, single channel
        image = np.load(os.path.join(cfg['data_dir'], v.file))
        # basic transforms for pred - faking RGB by repeating 3 times
        image = np.repeat(image[..., np.newaxis], 3, -1)
        image = Compose([ToPILImage(), 
                         Resize([224, 224]), ToTensor(),
                         Normalize(mean=cfg['norm_mean'],
                                             std=cfg['norm_sd'])])(image)
        image.requires_grad_()
        # none is for extras spot - put in real verison later if i have extras
        # need unsqueeze to fake batch of 1
        pred = model(image.unsqueeze(0), None)
        
        # direct copy from Medium article
        score_max_index = pred.argmax()
        score_max = pred[0,score_max_index]
        score_max.backward()
        saliency, _ = torch.max(image.grad.data.abs(),dim=0)
        
        # flipud makes image look right
        sal_out.append(np.flipud(saliency.numpy()))

    return(sal_out)