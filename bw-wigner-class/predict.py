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
import pandas as pd
import sklearn.metrics as met
import matplotlib.pyplot as plt

def predict(cfg, model, label_csv):
    
    device = cfg['device']
    # load model if pointing to a state file
    if isinstance(model, str):
        state = torch.load(open(model, 'rb'), map_location='cpu')
        model = BeakerNet(cfg)
        model.load_state_dict(state['model'])
    
    dataset = BWDataset(cfg, label_csv, 
                        Compose([ToPILImage(), Resize([224, 224]), ToTensor()]))
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg['batch_size'],
        num_workers=cfg['num_workers'],
        shuffle=False
        )
    
    my_dict = {'pred': [],
               'true': np.array(dataset.species)
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
            my_dict['pred'].extend(
                torch.argmax(prediction, dim=1).detach().to('cpu').numpy().tolist())
            pred_prob.append(
                torch.softmax(prediction, dim=1).detach().to('cpu').numpy())
            pb.update(1)
    
    pb.close()

    df = pd.read_csv(label_csv)
    # some files are NA bc I exported all w/o filtering, drop them now
    df = df[-np.isnan(df.wigMax)]
    df['pred'] = np.array(my_dict['pred'])
    df['true'] = my_dict['true']
    pred_prob = np.concatenate(pred_prob)
    prob_df = pd.DataFrame(pred_prob, columns=['p'+str(x) for x in range(cfg['num_classes'])])
    df = pd.concat([df.reset_index(drop=True), 
                    prob_df.reset_index(drop=True)],
                   axis = 1)
    return(df)   
    
def pred_plots(df, cfg, name):
    # two confusion matrices
    cm_true = met.confusion_matrix(df.true, df.pred, normalize='true')
    cmd_true = met.ConfusionMatrixDisplay(cm_true)
    cm_pred = met.confusion_matrix(df.true, df.pred, normalize='pred')
    cmd_pred = met.ConfusionMatrixDisplay(cm_pred)
    fig = plt.figure(figsize=(10, 5))
    ax1 = plt.subplot(1, 2, 1)
    ax1.set_title('Norm across TRUE (Recall)')
    cmd_true.plot(ax=ax1)
    ax2=plt.subplot(1, 2, 2)
    ax2.set_title('Norm across PRED (Precision)')
    cmd_pred.plot(ax=ax2)
    plt.savefig('ConfMats_'+name+'.png')
    # PR curve by species
    
    plt.figure(figsize=(5,5))
    cmap = plt.get_cmap('Set2')
    inv_sp = {cfg['sp_dict'][x]: x for x in cfg['sp_dict']}
    for i in range(5):
        # avg_prec[i] = met.average_precision_score((df.true.values == i).astype(int), probs[:, i])
        p, r, t = met.precision_recall_curve(df.true.values == i, df['p'+str(i)].values)
        plt.plot(p, r, color=cmap(i), label=inv_sp[i])
    # print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
    plt.legend()
    plt.savefig('PRCurve_'+name+'.png')
    
def main():
    # set up command line argument parser for cfg file
    parser = argparse.ArgumentParser(description='Predict yo BeakerNet CLICK CLICK BOIIII')
    parser.add_argument('--config', help='Path to config file', default='configs/bn1_resnet50.yaml')
    parser.add_argument('--model', help='Model state to predict with', default='model_states/0.pt')
    parser.add_argument('--name', help='Name to append to CSV', default='')
    args = parser.parse_args()
    
    # load config
    print(f'Using config "{args.config}"')
    cfg = yaml.safe_load(open(args.config, 'r'))
    suff = '_' + args.name + 'pred.csv'
    
    # do pred on train
    label_train = os.path.join(cfg['label_dir'], cfg['label_csv']['train'])
    pred_train = predict(cfg, args.model, label_train)
    pred_train.to_csv(re.sub('.csv', suff, label_train))
    pred_plots(pred_train, cfg, 'train')
    
    # do pred on val
    label_val = os.path.join(cfg['label_dir'], cfg['label_csv']['val'])
    pred_val = predict(cfg, args.model, label_val)
    pred_val.to_csv(re.sub('.csv', suff, label_val))
    pred_plots(pred_val, cfg, 'val')
    
    # only pred on test if we want to
    if cfg['pred_test']:
        label_test = os.path.join(cfg['label_dir'], cfg['label_csv']['test'])
        pred_test = predict(cfg, args.model, label_test)
        pred_test.to_csv(re.sub('.csv', suff, label_test))
    
    # cfg_base = re.sub('\\..*$', '', os.path.basename(args.config))
    # mod_base = re.sub('\\..*$', '', os.path.basename(args.model))
    #with open('preds_'+cfg_base+'_'+mod_base+'.txt', 'w') as file:
    #    file.write(json.dumps(pre_dict))
    # np.save('preds_'+cfg_base+'_'+mod_base, pre_dict['pred'])
    # np.save('true_'+cfg_base+'_'+mod_base, pre_dict['true'])
    # np.save('probs_'+cfg_base+'_'+mod_base, probs
    
if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()
