# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 16:53:06 2022

@author: tnsak
"""
from train import create_dataloader

def predict(cfg, model):
    
    device = cfg['device']
    # I might not want to do this...
    dataloader = create_dataloader(cfg, 'predict')
    # send to device and set to train mode
    model.to(device)
      
    for idx, (data, label) in enumerate(dataloader):
        # put on device for model speed
        data, label = data.to(device), label.to(device)
        # forward, beakernet!
        prediction = model(data)
        
        # what is proper way to accumulate these preds/probs
        pred_label = torch.argmax(prediction, dim=1)
        pred_prob = torch.softmax(prediction, dim=1)
        oa = torch.mean((pred_label == label).float())
        oa_total += oa.item()
  

    return(loss_total, oa_total)