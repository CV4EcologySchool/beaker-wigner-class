# -*- coding: utf-8 -*-
"""
Created on Wed Aug 10 16:53:06 2022

@author: tnsak
"""
from dataset import BWDataset
from model import BeakerNet
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, Normalize
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
from util import get_saliency

def predict(cfg, model, label_csv):
    
    device = cfg['device']
    # load model if pointing to a state file
    if isinstance(model, str):
        state = torch.load(open(model, 'rb'), map_location='cpu')
        model = BeakerNet(cfg)
        model.load_state_dict(state['model'])
    
    dataset = BWDataset(cfg, label_csv, 
                        Compose([ToPILImage(), 
                                 Resize([224, 224]), 
                                 ToTensor(),
                                 Normalize(mean=cfg['norm_mean'],
                                                     std=cfg['norm_sd'])]))
    
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
        for idx, (data, label, snr, extras) in enumerate(dataloader):
            # put on device for model speed
            data, label = data.to(device), label.to(device)
            if cfg['extra_params']:
                extras = extras.to(device)
            else:
                extras = None
            # forward, beakernet!
            prediction = model(data, extras)
            
            # what is proper way to accumulate these preds/probs
            my_dict['pred'].extend(
                torch.argmax(prediction, dim=1).detach().to('cpu').numpy().tolist())
            pred_prob.append(
                torch.softmax(prediction, dim=1).detach().to('cpu').numpy())
            pb.update(1)
    
    pb.close()

    df = pd.read_csv(label_csv)
    # some files are NA bc I exported all w/o filtering, drop them now
    for drop in cfg['check_na_col']:
        df = df[-np.isnan(df[drop])]
    df = df[df.snr > cfg['snr_filt_min']]
    
    df['pred'] = np.array(my_dict['pred'])
    df['true'] = my_dict['true']
    pred_prob = np.concatenate(pred_prob)
    prob_df = pd.DataFrame(pred_prob, columns=['p'+str(x) for x in range(cfg['num_classes'])])
    df = pd.concat([df.reset_index(drop=True), 
                    prob_df.reset_index(drop=True)],
                   axis = 1)
    return(df)   
    
def pred_plots(df, cfg, name, model):
    outdir = cfg['pred_dir']
    # two confusion matrices
    cm_true = met.confusion_matrix(df.true, df.pred, normalize='true')
    cmd_true = met.ConfusionMatrixDisplay(cm_true)
    cm_pred = met.confusion_matrix(df.true, df.pred, normalize='pred')
    cmd_pred = met.ConfusionMatrixDisplay(cm_pred)
    cm_none = met.confusion_matrix(df.true, df.pred, normalize=None)
    cmd_none = met.ConfusionMatrixDisplay(cm_none)
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(2, 2, 3)
    ax1.set_title('Norm across TRUE (Recall)')
    cmd_true.plot(ax=ax1)
    ax2=plt.subplot(2, 2, 4)
    ax2.set_title('Norm across PRED (Precision)')
    cmd_pred.plot(ax=ax2)
    ax3=plt.subplot(2, 2, 2)
    ax3.set_title('Norm across NONE')
    cmd_none.plot(ax=ax3)
    cl_rep = met.classification_report(df.true, df.pred)
    ax4 = plt.subplot(2, 2, 1)
    ax4.text(x=.08, y=.2, s=cl_rep)
    ax4.axis('off')
    plt.savefig(os.path.join(outdir, 'ConfMats_'+name+'.png'))
    
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
    plt.savefig(os.path.join(outdir, 'PRCurve_'+name+'.png'))
    
    # best and worst images
    top_tp, top_fp, top_fn = get_top_n(df, cfg['plot_top_n'])
    plot_top_n(top_tp, cfg,
               os.path.join(outdir, 'Top'+str(cfg['plot_top_n'])+'TP_'+name+'.png'),
               lab_true = True, title = 'Predicted vs True', sal=cfg['do_sal'], model=model)
    plot_top_n(top_fp, cfg,
               os.path.join(outdir, 'Top'+str(cfg['plot_top_n'])+'FP_'+name+'.png'),
               lab_true = True, title = 'Predicted vs True', sal=cfg['do_sal'], model=model)
    plot_top_n(top_fn, cfg,
            os.path.join(outdir, 'Top'+str(cfg['plot_top_n'])+'FN_'+name+'.png'),
               lab_true = False, title='True vs Predicted', sal=cfg['do_sal'], model=model)

def event_metrics(df, cfg, name):
    ev = df.groupby('station').agg(
        sump0 = ('p0', 'sum'),
        sump1 = ('p1', 'sum'),
        sump2 = ('p2', 'sum'),
        sump3 = ('p3', 'sum'),
        sump4 = ('p4', 'sum'),
        true = ('true', 'median')).reset_index()

    ev['pred'] = ev[['sump'+str(x) for x in range(5)]].apply(lambda x: np.argmax(x), axis=1)
    outdir = cfg['pred_dir']
    # two confusion matrices
    cm_true = met.confusion_matrix(ev.true, ev.pred, normalize='true')
    cmd_true = met.ConfusionMatrixDisplay(cm_true)
    cm_pred = met.confusion_matrix(ev.true, ev.pred, normalize='pred')
    cmd_pred = met.ConfusionMatrixDisplay(cm_pred)
    cm_none = met.confusion_matrix(ev.true, ev.pred, normalize=None)
    cmd_none = met.ConfusionMatrixDisplay(cm_none)
    fig = plt.figure(figsize=(10, 10))
    ax1 = plt.subplot(2, 2, 3)
    ax1.set_title('Norm across TRUE (Recall)')
    cmd_true.plot(ax=ax1)
    ax2=plt.subplot(2, 2, 4)
    ax2.set_title('Norm across PRED (Precision)')
    cmd_pred.plot(ax=ax2)
    ax3=plt.subplot(2, 2, 2)
    ax3.set_title('Norm across NONE')
    cmd_none.plot(ax=ax3)
    cl_rep = met.classification_report(ev.true, ev.pred)
    ax4 = plt.subplot(2, 2, 1)
    ax4.text(x=.08, y=.2, s=cl_rep)
    ax4.axis('off')
    plt.savefig(os.path.join(outdir, 'EventConfMats_'+name+'.png'))
    
def get_top_n(df, n_top=5):
    top_tp = []
    top_fp = []
    top_fn = []
    
    classes = np.sort(df.true.unique())
    for i in classes:
        df = df.sort_values(by='p'+str(i), ascending=False)
        top_tp.append(df[(df.true == i) & (df.pred == i)].head(n_top))
        top_fp.append(df[(df.true != i) & (df.pred == i)].head(n_top))
        others = ['p'+str(x) for x in classes[classes != i]]
        df['fn_sort'] = df[others].apply(max, axis=1)
        df = df.sort_values(by='fn_sort', ascending=False)
        top_fn.append(df[(df.true == i) & (df.pred != i)].head(n_top))
    
    return(top_tp, top_fp, top_fn)

def plot_top_n(df, cfg, name='TopN.png', lab_true=False, title='', sal=False, model=None):
    n_top = cfg['plot_top_n']
    if type(df) != list:
        df = get_top_n(df, n_top)
    
    inv_sp = {cfg['sp_dict'][x]: x for x in cfg['sp_dict']}
    data_dir = cfg['data_dir']
            
    #classes = np.sort(np.array([x.true.values[0] for x in df if len(x) > 0]))
    classes = range(cfg['num_classes'])
    fsize = 1.5
    # plt.figure(figsize=(fsize * len(classes), fsize * n_top))
    fig, ax = plt.subplots(len(classes), n_top*(1+sal), 
                           figsize=(fsize*len(classes)*(1+sal)+.5, fsize * n_top))
    
    for i, tf in enumerate(df):
        if sal:
            sal_data = get_saliency(tf, cfg, model)
            # fix all Js here
        for j in range(n_top):
            use_j = j * (1+sal)
            ax[i, use_j].axis('off')
            if sal:
                ax[i, use_j+1].axis('off')
            if j == 0:
                ax[i, j].set_ylabel(inv_sp[i])
            if j >= tf.shape[0]:
                continue
            imfile = os.path.join(data_dir, tf.file.values[j])
            ax[i, use_j].imshow(np.flipud(np.load(imfile)))
            sp_lab = tf.true.values[j] if lab_true else tf.pred.values[j]
            ax[i, use_j].text(x=0, y=1, s=inv_sp[sp_lab], c='white', va='top')
            ax[i, use_j].text(x=0, y=125, s='SNR '+str(round(tf.snr.values[j])), c='white', fontsize=8)
            if sal:
                ax[i,use_j+1].imshow(sal_data[j], cmap=plt.cm.hot)
            # and then set label label for all to show misclass
            # ax[i, j].set_title(inv_sp[i])
    fig.tight_layout(pad=.01)
    if len(title):    
        fig.suptitle(title)
        fig.subplots_adjust(top=.9)
    fig.savefig(name)
    
def do_pred_work(cfg, args, split='train', pred_df=None):
    suff = '_' + args.name + 'pred.csv'
    outdir = cfg['pred_dir']
    os.makedirs(outdir, exist_ok=True)
    
    if pred_df is None:
        label_csv = os.path.join(cfg['label_dir'], cfg['label_csv'][split])
        pred_df = predict(cfg, args.model, label_csv)
        out_csv = os.path.join(outdir, 
                               re.sub('.csv', suff, os.path.basename(label_csv)))
        pred_df.to_csv(out_csv, index=False)
    else:
        pred_df = pd.read_csv(pred_df)
        cfg['do_sal'] = False
        
    pred_plots(pred_df, cfg, args.name+'_'+split, args.model)
    event_metrics(pred_df, cfg, args.name+'_'+split)
    
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
    # suff = '_' + args.name + 'pred.csv'
    # outdir = cfg['pred_dir']
    # os.makedirs(outdir, exist_ok=True) 
    if '.pt' in args.model:
        do_list = ['train', 'val', 'test'] if cfg['pred_test'] else ['train', 'val']
        for do in do_list:
            do_pred_work(cfg, args, split=do)
    elif '.csv' in args.model:
        do_pred_work(cfg, args, split='', pred_df=args.model)
               
if __name__ == '__main__':
    # This block only gets executed if you call the "train.py" script directly
    # (i.e., "python ct_classifier/train.py").
    main()
