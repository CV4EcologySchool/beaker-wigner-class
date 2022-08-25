# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 16:09:25 2022

@author: tnsak
"""
import os
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, ToTensor, ToPILImage, Grayscale
from torchvision.utils import make_grid
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml

class BWDataset(Dataset):
    
    def __init__(self, cfg, label_csv, transform=Compose([ToPILImage(), Resize([224, 224]), ToTensor()])):
        '''
        Constructor for data loader, just loading in from file locs
        Also need to specify base dir of file paths since different
        from when i created them in R. 
        Not sure if this should go in a config file or if its okay 
        as a param in the data loade
        '''
        df = pd.read_csv(label_csv)
        # some files are NA bc I exported all w/o filtering, drop them now
        for drop in cfg['check_na_col']:
            df = df[-np.isnan(df[drop])]
        df = df[df.snr > cfg['snr_filt_min']]    
        self.file = [os.path.join(cfg['data_dir'], x) for x in list(df.file)]
        self.csvname = label_csv
        self.species = [cfg['sp_dict'][x] for x in df.species.tolist()]
        self.sp_dict = cfg['sp_dict']
        self.transform = transform
        self.wigMax = np.log(df.wigMax.values)
        self.wigMax -= min(self.wigMax)
        self.wigMax /= max(self.wigMax)
        self.wigMax = self.wigMax[:, np.newaxis]
        snr_par = cfg['snr_scale_params']
        self.snr_scale = df.snr.values / snr_par['max_val'] \
            * (1-snr_par['min_prob']) + snr_par['min_prob']
        self.snr_scale[self.snr_scale > 1] = 1
    
    def __len__(self):
        return len(self.file)
    
    def __getitem__(self, ix):
        image = np.load(self.file[ix])
        # image = image - (np.median(image)-127)
        # just repeating to fake RGB?
        image = np.repeat(image[..., np.newaxis], 3, -1)
        # print(image.shape)
        label = self.species[ix]
        extras = torch.tensor(self.wigMax[ix])
        # extras = torch.hstack([extras, extras])
        # make tensor
        if(self.transform):
            image = self.transform(image)
        # print(image.shape)
        '''
        very unclear how I need to modify image input from uint8 nparray
        do lables need to be one-hot encoding??
        Models are expecting RGB? But I have gray?
        Some SE answers say just repeat the channel 3 times
        '''
        snr = self.snr_scale[ix]
        
        return image, label, snr, extras
    
    def showImg(self, ix):
        image, label, snr, extras = self[ix]
        # image = np.moveaxis(image.numpy()*255, 0, -1).astype(np.uint8)[:,:,0]
        image = Grayscale(1)(image).numpy().squeeze(0)
        plt.imshow(np.flipud(image))
        plt.title('Species: ' + str(label) + ' File: ' + os.path.basename(self.file[ix])) 
        plt.yticks(np.linspace(0, 128, 5), np.linspace(96, 0, 5))
        plt.ylabel('Frequency (kHz)')
        plt.show()
        
    def multiImg(self, ix, ncol=5):
        '''
        first attempt
        '''
        nrow = int(np.ceil(len(ix)/ncol))
        fig, ax = plt.subplots(nrow, ncol)
        # fig = plt.figure()
        # gs = fig.add_gridspec(int(np.ceil(len(ix)/ncol)), ncol, hspace=0, wspace=0)
        # ax = gs.subplots(sharex=True, sharey=True)
        for i, v in enumerate(ix):
            im, lab, snr, extra = self[v]
            im = np.moveaxis(im.numpy()*255, 0, -1).astype(np.uint8)[:,:,0]
            im = np.flip(im)
            row = int(np.floor(i / ncol))
            col = i % ncol
            # print('Row ' + str(row) + ' Col ' + str(col))
            ax[row, col].imshow(im)
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            # ax[row, col].margins(0.01, tight=True)
        # fig.tight_layout(pad=.05)
        fig.subplots_adjust(bottom=0, top=1, left=0, right=1)
        '''
        second attempt with torchvision grid
        '''
        
        
