# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 16:53:06 2022

@author: tnsak
"""
from dataset import BWDataset
from model import BeakerNet
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
import os
from torch.utils.data import DataLoader
import torch
import argparse
import yaml
import re
import json
from tqdm import trange
import numpy as np

def predict(cfg, model):
    
    device = cfg['device']
    # load model if pointing to a state file
    if isinstance(model, str):
        state = torch.load(open(model, 'rb'), map_location='cpu')
        model = BeakerNet(cfg)
        model.load_state_dict(state['model'])
    
    label_csv = os.path.join(cfg['label_dir'], cfg['label_csv']['predict'])
    
    dataset = BWDataset(cfg, label_csv, 
                        Compose([ToPILImage(), Resize([224, 224]), ToTensor()]))
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        shuffle=False
        )
    
    my_dict = {'pred_label': [],
               'true_label': dataset.species
        }
    pred_prob = []
    # send to device and set to train mode
    model.to(device)
    model.eval()
    pb = trange(len(dataloader))
    with torch.no_grad():
        for idx, (data, label) in enumerate(dataloader):
            # put on device for model speed
            data, label = data.to(device), label.to(device)
            # forward, beakernet!
            prediction = model(data)
            
            # what is proper way to accumulate these preds/probs
            my_dict['pred_label'].append(
                torch.argmax(prediction, dim=1).detach().to('cpu').numpy().tolist())
            pred_prob.append(
                torch.softmax(prediction, dim=1).detach().to('cpu').numpy())
            pb.update(1)
    
    pb.close()
    # print(my_dict['pred_label'][0])
    # print(type(my_dict['pred_label'][0]))
    # print(pred_prob[0])
    # print(type(pred_prob[0]))
    
    return(my_dict, pred_prob)

def main():
    # set up command line argument parser for cfg file
    parser = argparse.ArgumentParser(description='Predict yo BeakerNet CLICK CLICK BOIIII')
    parser.add_argument('--config', help='Path to config file', default='configs/bn1_resnet50.yaml')
    parser.add_argument('--model', help='Model state to predict with', default='model_states/0.pt')
    
    args = parser.parse_args()
    
    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
    pre_dict, probs = predict(cfg, args.model)
    
    cfg_base = re.sub('\\..*$', '', os.path.basename(args.config))
    mod_base = re.sub('\\..*$', '', os.path.basename(args.model))
    with open('preds_'+cfg_base+'_'+mod_base+'.txt', 'w') as file:
        file.write(json.dumps(pre_dict))
        
    np.save('probs_'+cfg_base+'_'+mod_base, probs)
    
if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()