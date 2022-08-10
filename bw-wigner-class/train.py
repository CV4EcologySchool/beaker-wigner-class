# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 16:09:47 2022

@author: tnsak
"""
from dataset import BWDataset
from model import BeakerNet
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
import os
import torch.nn as nn
from tqdm import trange
import torch

def create_dataloader(cfg, transform, split='train'):
    '''
    create a dataloader for a dataset instance
    '''
    label_csv = os.path.join(cfg['label_dir'], cfg['label_csv'][split])
    dataset = BWDataset(cfg, label_csv, transform)
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        shuffle=True
        )
    return dataloader

def train(cfg, dataloader, model, optimizer):
    device = cfg['device']
    # send to device and set to train mode
    model.to(device)
    model.train()
    
    # define loss function - MAY CHANGE LATER
    criterion = nn.CrossEntropyLoss()
    
    # init running averages
    loss_total, oa_total = 0.0, 0.0
    
    pb = trange(len(dataloader))
    
    for idx, (data, label) in enumerate(dataloader):
        # put on device for model speed
        data, label = data.to(device), label.to(device)
        # forward, beakernet!
        prediction = model(data)
        # have to reste grads
        optimizer.zero_grad()
        # calc loss and full send back
        loss = criterion(prediction, label)
        loss.backwards()
        
        optimizer.step()
        
        loss_total += loss.item()
        
        pred_label = torch.argmax(prediction, dim=1)
        oa = torch.mean((pred_label == label).float())
        oa_total += oa.item()
        
        pb.set_description(
            '[Train] Loss {:2f}; OA {:2f}%'.format(
                loss_total/(idx + 1),
                100*oa_total/(idx + 1)
                )
            )
        pb.update(1)
    
    pb.close()
    loss_total /= len(dataloader)
    oa_total /= len(dataloader)
    
    return(loss_total, oa_total)
    
    
def main():
    
    trans_dict = {
        'train': Compose([ToPILImage(), Resize([224, 224]), ToTensor()]),
        'test': Compose([ToPILImage(), Resize([224, 224]), ToTensor()])
                  }
    heydog = 'dawgggg'