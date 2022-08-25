# -*- coding: utf-8 -*-
"""
Created on Wed Aug 24 14:43:48 2022

@author: tnsak
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage
from tqdm import trange
import torch
from model import BeakerNet

def plot_all_event(df, cfg, ncol=5, by='snr', name='', outdir='.', sal=False):
    
    # df = df[df.true == sp]
    df = df.sort_values(by=by, ascending=False)
    df_list = [df.iloc[range(x, min(x+ncol, len(df)))] for x in range(0, len(df), ncol)]
        
    nrow = len(df_list)
    
    inv_sp = {cfg['sp_dict'][x]: x for x in cfg['sp_dict']}
    # data_dir = cfg['data_dir']
    data_dir = './data'
            
    fsize = 1.5
    
    fig, ax = plt.subplots(nrow, ncol*(1+sal), figsize=(fsize*ncol*(1+sal), fsize * nrow), squeeze=False)
    
    for i, tf in enumerate(df_list):
        # print('i'+str(i))
        for j in range(ncol):
            # print('j'+str(j))
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            if j == 0:
                ax[i, j].set_ylabel(inv_sp[tf.true.values[j]])
            if j >= tf.shape[0]:
                ax[i, j].axis('off')
                continue

            imfile = os.path.join(data_dir, tf.file.values[j])
            ax[i, j].imshow(np.flipud(np.load(imfile)))
            sp_lab = tf.pred.values[j]
            sp_col = 'white' if tf.pred.values[j] == tf.true.values[j] else 'red'
            ax[i, j].text(x=0, y=1, s=inv_sp[sp_lab], c=sp_col, va='top')
            ax[i, j].text(x=0, y=125, s='SNR '+str(round(tf.snr.values[j])), c='white', fontsize=8)
            # and then set label label for all to show misclass
            # ax[i, j].set_title(inv_sp[i])
    fig.tight_layout(pad=.01)
    if len(name):    
        fig.suptitle(name)
        fig.subplots_adjust(top=.9)
    fig.savefig(os.path.join(outdir, name + '_by-' + by + '.png'))
    # fig.show()
    plt.close(fig)

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
        image = np.load(os.path.join('./data/', v.file))
        # just repeating to fake RGB?
        image = np.repeat(image[..., np.newaxis], 3, -1)
        image = Compose([ToPILImage(), Resize([224, 224]), ToTensor()])(image)
        # print(image.shape)
        image.requires_grad_()
        pred = model(image.unsqueeze(0), None)
        
        score_max_index = pred.argmax()
        score_max = pred[0,score_max_index]
        score_max.backward()
        # print(image.grad.data.abs().shape)
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

def plot_saliency(df, cfg, model):
    if isinstance(model, str):
        state = torch.load(open(model, 'rb'), map_location='cpu')
        model = BeakerNet(cfg)
        model.load_state_dict(state['model'])
    
    model.eval()
    fsize=1.5
    fig, ax = plt.subplots(len(df), 2, figsize=(fsize*2, fsize*len(df)))
    for i, v in enumerate(df.itertuples(index=False)):
        image = np.load(v.file)
        # just repeating to fake RGB?
        image = np.repeat(image[..., np.newaxis], 3, -1)
        image = Compose([ToPILImage(), Resize([224, 224]), ToTensor()])(image)
        image.requires_grad_()
        pred = model(image)
        
        score_max_index = pred.argmax()
        score_max = pred[0,score_max_index]
        score_max.backward()

        saliency, _ = torch.max(image.grad.data.abs(),dim=1)
        # saliency[0] is the shit
        ax[i,0].imshow(image)
        ax[i,1].imshow(saliency[0])
        
    
def main():
    csv_dict = {'train': 'export_preds_snrfilt5/PASCAL_BW_Train_R18SNRFilt5v31pred.csv',
                'val': 'export_preds_snrfilt5/PASCAL_BW_Val_R18SNRFilt5v31pred.csv',
                'test': 'export_preds_snrfilt5/PASCAL_BW_Test_R18SNRFilt5v31pred.csv'}
    # csv_list = csv_list[1]
    sp_do = [3]
    cfg = yaml.safe_load(open('configs/bn2_resnet18.yaml', 'r'))
    outdir = 'event_plots'
    for split in csv_dict:
        df = pd.read_csv(csv_dict[split])
        df = df.query('true in @sp_do')
        pb = trange(len(df.station.unique()))
        for st, st_df in df.groupby('station'):
            sp = str(st_df.true.values[0])
            plot_all_event(st_df, cfg, ncol=5, by='p'+sp, name=split+'_'+sp+'_'+st, outdir=outdir)
            pb.update(1)
        pb.close()
            
if __name__ == '__main__':
    main()