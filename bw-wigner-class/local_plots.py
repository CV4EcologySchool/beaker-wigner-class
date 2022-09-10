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
import numbers
#%%
def plot_all_pred(df, cfg, ncol=5, sort='', asc=False, name='',
                  data_dir=None, out_dir='.', sal=False, nmax=100):
    
    # df = df[df.true == sp]
    if sort:
        df = df.sort_values(by=sort, ascending=asc)
    # df_list = [df.iloc[range(x, min(x+ncol, len(df)))] for x in range(0, len(df), ncol)]
    if len(df) > nmax:
        df = df[:nmax]
    
    nrow = len(df) // ncol + (len(df) % ncol > 0)
    
    inv_sp = {cfg['sp_dict'][x]: x for x in cfg['sp_dict']}
    if data_dir is None:
        data_dir = cfg['data_dir']
    # data_dir = './data'
            
    fsize = 1.5
    
    fig, ax = plt.subplots(nrow, ncol*(1+sal), figsize=(fsize*ncol*(1+sal), fsize * nrow), squeeze=False)

    # for i, tf in enumerate(df_list):
    for i in range(nrow):
        # print('i'+str(i))
        for j in range(ncol):
            # print('j'+str(j))
            ix = i*ncol + j
            ax[i, j].set_xticks([])
            ax[i, j].set_yticks([])
            if ix >= len(df):
                ax[i, j].axis('off')
                continue
            if j == 0:
                ax[i, j].set_ylabel(inv_sp[df.true.values[ix]])

            imfile = os.path.join(data_dir, df.file.values[ix])
            ax[i, j].imshow(np.flipud(np.load(imfile)))
            sp_lab = df.pred.values[ix]
            sp_col = 'white' if df.pred.values[ix] == df.true.values[ix] else 'red'
            # lab = inv_sp[sp_lab]+df.file.str.replace('.*C([12]).npy$', '\\1').values[ix]
            lab = inv_sp[sp_lab]+str(round(df.snr.values[ix]))
            ax[i, j].text(x=0, y=1, s=lab, c=sp_col, va='top')
            if sort:
                by_lab = df[sort].values[ix]
                if isinstance(by_lab, numbers.Number):
                    # round based on scale of input, 2 if <1 1 if <10
                    rnd = 0 + (abs(by_lab) < 1) + (abs(by_lab) < 10)
                    by_lab = round(by_lab, rnd)
                    ax[i, j].text(x=0, y=125, s=sort+' '+str(by_lab), c='white', fontsize=8)
            # and then set label label for all to show misclass
            # ax[i, j].set_title(inv_sp[i])
    fig.tight_layout(pad=.01)
    if len(name):    
        fig.suptitle(name)
        fig.subplots_adjust(top=.9)
    fig.savefig(os.path.join(out_dir, name + '_by-' + sort + '.png'))
    # fig.show()
    plt.close(fig)
    
#%%
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