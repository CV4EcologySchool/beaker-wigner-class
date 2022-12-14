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
from pytorch_grad_cam import DeepFeatureFactorization, GradCAM
from pytorch_grad_cam.utils.image import show_factorization_on_image, show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import re
import pandas as pd
#%%
def parse_file_name(x, what='UID'):
    if what == 'UID':
        patt = '.*_OE[0-9]{1,5}_([0-9]{6,16})_C[12].npy'
    elif what == 'channel':
        patt = '.*_C([12]).npy'
        
    if type(x) == 'str':
        x = os.path.basename(x)
        return re.sub(patt, '\\1', x)
    return x.file.str.replace(patt, '\\1')
    
#%%
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
#%%
class CrossEntSNR(nn.CrossEntropyLoss):
    def __init__(self, weight):
        super().__init__(weight=weight)
    
    def forward(self, x, target, snr_scale):
        ce = super().forward(x, target)
        ce = ce * snr_scale
        return ce.mean()

class MyCE(nn.Module):
    def __init__(self, weight = None):
        super().__init__()
        self.weight=weight
        self.eps = 1e-7
    
    def forward(self, inputs, targets):
        batch = inputs.size()[0]
        inputs = torch.softmax(inputs, 1)
        # print(inputs[range(batch),targets])
    
        loss = - torch.log(inputs[range(batch), targets] + self.eps)
        # print(loss)
        if self.weight is not None:
            # print(self.weight[targets])
            loss = loss * self.weight[targets] 
            return torch.sum(loss)/ self.weight[targets].sum()
        else:
            return torch.sum(loss) / batch

class MySelCE(nn.Module):
    def __init__(self, weight=None, coverage=.8, lam=32):
        super().__init__()
        self.weight = weight
        self.eps = 1e-7
        self.lam = lam
        self.coverage = coverage
    
    def forward(self, inputs, targets):
        batch = inputs.size()[0]
        device = inputs.device
        # compute cross ent loss for preds scaled by selector prob (last col)
        # inputs[:, -1] is q from paper - selection prob
        in_pred = torch.softmax(inputs[:, :-1], 1)
        ce_loss = - torch.log(in_pred[range(batch), targets]) * inputs[:, -1]
        if self.weight is not None:
            ce_loss = ce_loss * self.weight[targets]
            ce_loss = torch.sum(ce_loss) / self.weight[targets].sum()
        else:
            ce_loss = torch.sum(ce_loss) / batch
        # compute power loss for difference from desired coverage
        if self.weight is not None:
            emp_cov = torch.sum(inputs[:, -1] * self.weight[targets]) / self.weight[targets].sum()
        else:
            emp_cov = torch.mean(inputs[:, -1])
        lam_loss = self.lam * torch.pow(torch.maximum(self.coverage - emp_cov, torch.Tensor([0]).to(device)), 2)
        # print(ce_loss/emp_cov)
        # print(emp_cov)
        # print(lam_loss)
        return ce_loss/ emp_cov + lam_loss
    
#%%  
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

def create_labels(concept_scores, top_k=2):
    labels = ['ZC', 'BB', 'MS', 'BW43', 'BW37V']
    concept_categories = np.argsort(concept_scores, axis=1)[:, ::-1][:, :top_k]
    concept_labels_topk = []
    for concept_index in range(concept_categories.shape[0]):
        categories = concept_categories[concept_index, :]    
        concept_labels = []
        for category in categories:
            score = concept_scores[concept_index, category]
            label = f"{labels[category].split(',')[0]}:{score:.2f}"
            concept_labels.append(label)
        concept_labels_topk.append("\n".join(concept_labels))
    return concept_labels_topk

def get_img(file, cfg):
    """A function that gets a URL of an image, 
    and returns a numpy image and a preprocessed
    torch tensor ready to pass to the model """
    img = np.load(file)
    img = np.repeat(img[..., np.newaxis], 3, -1)
    input_tensor = Compose([ToPILImage(), 
                     Resize([224, 224]), ToTensor()])(img)
    img = (input_tensor.permute(1,2,0).numpy()*255).astype(np.uint8)
    
    rgb_img_float = np.float32(img)/255
    input_tensor = Normalize(mean=cfg['norm_mean'],
                        std=cfg['norm_sd'])(input_tensor)
    return img, rgb_img_float, input_tensor

def do_dff(model, file, cfg, layer=-1, top_k=2):
    img, rgb_img_float, input_tensor = get_img(file, cfg)
    classifier = model.classifier
    n_components = cfg['n_dff']
    dff = DeepFeatureFactorization(model=model,
                                   target_layer=model.feature_extractor.layer4[layer], 
                                   computation_on_concepts=classifier)
    concepts, batch_explanations, concept_outputs = dff(input_tensor.unsqueeze(0), n_components)
    
    concept_outputs = torch.softmax(torch.from_numpy(concept_outputs), axis=-1).numpy()    
    concept_label_strings = create_labels(concept_outputs, top_k=top_k)
    visualization = show_factorization_on_image(rgb_img_float, 
                                                batch_explanations[0],
                                                image_weight=0.5,
                                                concept_labels=concept_label_strings)
    # result = np.hstack((img, visualization))
    # print(img.shape)
    # result = visualization
    visualization[:, 0:img.shape[1], :] = np.flipud(visualization[:, 0:img.shape[1], :])
    # Just for the jupyter notebook, so the large images won't weight a lot:
    # if result.shape[0] > 500:
    #     result = cv2.resize(result, (result.shape[1]//4, result.shape[0]//4))
    
    return visualization

def do_gradcam(model, file, layer, cfg, norm=True, targets=None):
    if isinstance(model, str):
        state = torch.load(open(model, 'rb'), map_location='cpu')
        model = BeakerNet(cfg)
        model.load_state_dict(state['model'])
    
    if type(layer) is int:
        layer = [layer]
    cam = GradCAM(model=model, 
                  target_layers = [model.feature_extractor.layer4[x] for x in layer])
    image = np.load(file)
    image = np.repeat(image[..., np.newaxis], 3, -1)
    image = Compose([ToPILImage(), 
                     Resize([224, 224]), 
                     ToTensor()])(image)
    if norm:
        image = Normalize(mean=cfg['norm_mean'], std=cfg['norm_sd'])(image)
    # targets = None
    if targets is not None:
        targets = [ClassifierOutputTarget(x) for x in targets]
    cam_im = cam(image.unsqueeze(0), aug_smooth=True, eigen_smooth=True,targets=targets)
    
    if norm:
        image = Compose([Normalize(mean=[0,0,0], 
                                   std=[1/x for x in cfg['norm_sd']]),
                         Normalize(mean=[-x for x in cfg['norm_mean']], 
                                   std=[1,1,1])])(image)
    return np.flipud(show_cam_on_image(image.permute(1,2,0).numpy(),
                      cam_im.squeeze(0), use_rgb=True))