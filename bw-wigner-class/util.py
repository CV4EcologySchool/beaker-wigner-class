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
from torchvision.transforms import Compose, Resize, ToPILImage, ToTensor
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

# df from predictions, model .pt
def get_saliency(df, cfg, model):
    if isinstance(model, str):
        state = torch.load(open(model, 'rb'), map_location='cpu')
        model = BeakerNet(cfg)
        model.load_state_dict(state['model'])
    
    model.eval()
    fsize=1.5
    sal_out = []
    # fig, ax = plt.subplots(len(df), 2, figsize=(fsize*2, fsize*len(df)), squeeze=False)
    for i, v in enumerate(df.itertuples(index=False)):
        image = np.load(os.path.join(cfg['data_dir'], v.file))
        # basic transforms for pred
        image = np.repeat(image[..., np.newaxis], 3, -1)
        image = Compose([ToPILImage(), Resize([224, 224]), ToTensor()])(image)
        # 
        image.requires_grad_()
        # none is for extras
        pred = model(image.unsqueeze(0), None)
        
        score_max_index = pred.argmax()
        score_max = pred[0,score_max_index]
        score_max.backward()
        
        saliency, _ = torch.max(image.grad.data.abs(),dim=0)
        sal_out.append(np.flipud(saliency.numpy()))
        # saliency[0] is the shit
    #     image = np.moveaxis(image.detach().numpy()*255, 0, -1).astype(np.uint8)[:,:,0]
    #     ax[i,0].imshow(np.flipud(image))
    #     ax[i,0].axis('off')
    #     print(saliency.shape)
    #     ax[i,1].imshow(np.flipud(saliency.numpy()), cmap=plt.cm.hot)
    #     ax[i,1].axis('off')
    # fig.tight_layout(pad=.01)
    # fig.show()
    return(sal_out)